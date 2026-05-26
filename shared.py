
"""Общие сущности, формулы и ML-утилиты для ИС анализа стресса."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import hmac
import json
import math
import os
import re
import statistics
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_NAME = "ИС анализа стресса игроков"
DB_NAME = "stress_is.sqlite3"
DATA_DIR = Path("data")
MEDIA_DIR = DATA_DIR / "media"
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"

SECRET_KEY = os.environ.get("STRESS_IS_SECRET", "change-me-in-production")
JWT_ALG = "HS256"
TOKEN_EXPIRE_HOURS = 24 * 14

DISEASE_CODES = {
    "healthy": "З",
    "asthma": "Л",
    "lung": "Л",
    "heart": "С",
    "cardio": "С",
    "other": "О",
}

ROLE_PLAYER = "player"
ROLE_RESEARCHER = "researcher"
ROLES = {ROLE_PLAYER, ROLE_RESEARCHER}

GENDER_CHOICES = ("мужской", "женский")
GENDER_DEFAULT = "не указан"

DEFAULT_WEIGHTS = {
    "heart": 0.40,
    "telemetry": 0.35,
    "ml": 0.25,
}

RULE_CALIBRATION_PATH = MODELS_DIR / "rule_calibration.json"
TELEMETRY_CALIBRATION_PATH = MODELS_DIR / "telemetry_calibration.json"

DEFAULT_TELEMETRY_WEIGHTS = {
    "motor": 0.30,
    "control": 0.24,
    "cognitive": 0.18,
    "behavioral": 0.28,
}

DEFAULT_TELEMETRY_REFERENCE = {
    "mouse_speed_mean": 420.0,
    "mouse_speed_std": 160.0,
    "mouse_speed_max": 760.0,
    "accel_mean": 1200.0,
    "accel_max": 2400.0,
    "jerk_mean": 2800.0,
    "motion_turn_rate": 0.20,
    "path_efficiency": 0.90,
    "motion_straightness": 0.88,
    "pause_count": 0.06,
    "key_hold_std": 0.16,
    "error_rate": 0.03,
    "backspace_count": 0.02,
    "escape_count": 0.01,
    "scroll_count": 0.04,
    "mouse_reversal_rate": 0.06,
    "mouse_spike_rate": 0.02,
    "double_click_rate": 0.02,
    "mouse_move_rate": 0.70,
}

DEFAULT_TELEMETRY_SCALES = {
    "mouse_speed_mean": 220.0,
    "mouse_speed_std": 120.0,
    "mouse_speed_max": 320.0,
    "accel_mean": 1100.0,
    "accel_max": 1600.0,
    "jerk_mean": 2200.0,
    "motion_turn_rate": 0.18,
    "path_efficiency": 0.12,
    "motion_straightness": 0.14,
    "pause_count": 0.08,
    "key_hold_std": 0.10,
    "error_rate": 0.05,
    "backspace_count": 0.03,
    "escape_count": 0.02,
    "scroll_count": 0.04,
    "mouse_reversal_rate": 0.08,
    "mouse_spike_rate": 0.03,
    "double_click_rate": 0.02,
    "mouse_move_rate": 0.40,
}

DISEASE_GROUP_CHOICES = ("healthy", "asthma", "heart")
DISEASE_GROUP_LABELS = {"healthy": "Здоровые", "asthma": "Астма/лёгкие", "heart": "Сердечные"}

HEART_LOW = 0.35
HEART_HIGH = 0.70
TEL_LOW = 0.35
TEL_HIGH = 0.70

# Session-level adaptive baseline for motion-derived telemetry.
_TELEMETRY_BASELINE_LOCK = threading.Lock()
_TELEMETRY_BASELINE = {
    "motion_speed_p90": 900.0,
    "motion_jerk_p90": 2600.0,
    "motion_accel_p90": 3600.0,
    "motion_turn_rate": 0.55,
    "path_efficiency": 0.90,
    "pause_count": 0.08,
    "error_rate": 0.02,
    "mouse_reversal_rate": 0.08,
    "mouse_spike_rate": 0.03,
}


def ensure_dirs() -> None:
    for p in (DATA_DIR, MEDIA_DIR, MODELS_DIR, CACHE_DIR):
        p.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s\-а-яА-ЯёЁ]", "", value, flags=re.UNICODE)
    value = re.sub(r"[\s\-]+", "_", value, flags=re.UNICODE)
    return value.strip("_")[:80] or "item"


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def movement_reversal(
    prev_dx: float,
    prev_dy: float,
    dx: float,
    dy: float,
    *,
    min_mag: float = 8.0,
    cosine_threshold: float = -0.35,
) -> bool:
    """Возвращает True, если текущее движение заметно развернулось относительно предыдущего.

    Ранее реверсом считалось почти любое движение по разным осям, из-за чего
    обычная навигация в игре давала ложные всплески телеметрии.
    """
    prev_mag = math.hypot(prev_dx, prev_dy)
    curr_mag = math.hypot(dx, dy)
    if prev_mag < min_mag or curr_mag < min_mag:
        return False
    denom = prev_mag * curr_mag
    if denom <= 1e-12:
        return False
    cosine = (prev_dx * dx + prev_dy * dy) / denom
    return cosine <= cosine_threshold

def _normalize_weight_map(weights: Dict[str, float]) -> Dict[str, float]:
    cleaned = {k: max(0.0, float(weights.get(k, 0.0))) for k in ("heart", "telemetry", "ml")}
    total = sum(cleaned.values())
    if total <= 1e-12:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in cleaned.items()}


def _threshold_from_scores(y_true: Sequence[int], scores: Sequence[float]) -> float:
    y = [1 if int(v) else 0 for v in y_true]
    s = [clip01(float(v)) for v in scores]
    if not y or len(set(y)) < 2:
        return 0.5

    best_t = 0.5
    best_f1 = -1.0
    for i in range(5, 96):
        t = i / 100.0
        tp = fp = fn = 0
        for yy, ss in zip(y, s):
            pred = 1 if ss >= t else 0
            if pred and yy:
                tp += 1
            elif pred and not yy:
                fp += 1
            elif (not pred) and yy:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t


def load_rule_calibration(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    baseline = default or {"weights": dict(DEFAULT_WEIGHTS), "threshold": 0.5, "updated_at": now_iso(), "calibrated": False}
    try:
        if RULE_CALIBRATION_PATH.exists():
            data = safe_json_loads(RULE_CALIBRATION_PATH.read_text(encoding="utf-8"), baseline)
            if isinstance(data, dict) and isinstance(data.get("weights"), dict):
                data["weights"] = _normalize_weight_map(data["weights"])
                data.setdefault("threshold", 0.5)
                data.setdefault("calibrated", False)
                return data
    except Exception:
        pass
    baseline["weights"] = _normalize_weight_map(baseline.get("weights", {}))
    return baseline


def save_rule_calibration(data: Dict[str, Any]) -> None:
    RULE_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["weights"] = _normalize_weight_map(payload.get("weights", {}))
    if "threshold" not in payload:
        payload["threshold"] = 0.5
    RULE_CALIBRATION_PATH.write_text(safe_json_dumps(payload), encoding="utf-8")


def _normalize_telemetry_group_weights(weights: Dict[str, float]) -> Dict[str, float]:
    cleaned = {k: max(0.0, float(weights.get(k, 0.0))) for k in DEFAULT_TELEMETRY_WEIGHTS}
    total = sum(cleaned.values())
    if total <= 1e-12:
        return dict(DEFAULT_TELEMETRY_WEIGHTS)
    return {k: v / total for k, v in cleaned.items()}


def load_telemetry_calibration(default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    baseline = default or {
        "weights": dict(DEFAULT_TELEMETRY_WEIGHTS),
        "reference": dict(DEFAULT_TELEMETRY_REFERENCE),
        "scales": dict(DEFAULT_TELEMETRY_SCALES),
        "idle_floor": 0.14,
        "warmup_samples": 6,
        "updated_at": now_iso(),
        "calibrated": False,
    }
    try:
        if TELEMETRY_CALIBRATION_PATH.exists():
            data = safe_json_loads(TELEMETRY_CALIBRATION_PATH.read_text(encoding="utf-8"), baseline)
            if isinstance(data, dict):
                if isinstance(data.get("weights"), dict):
                    data["weights"] = _normalize_telemetry_group_weights(data["weights"])
                else:
                    data["weights"] = dict(DEFAULT_TELEMETRY_WEIGHTS)
                if not isinstance(data.get("reference"), dict):
                    data["reference"] = dict(DEFAULT_TELEMETRY_REFERENCE)
                if not isinstance(data.get("scales"), dict):
                    data["scales"] = dict(DEFAULT_TELEMETRY_SCALES)
                data.setdefault("idle_floor", 0.14)
                data.setdefault("warmup_samples", 6)
                data.setdefault("calibrated", False)
                return data
    except Exception:
        pass
    baseline["weights"] = _normalize_telemetry_group_weights(baseline.get("weights", {}))
    return baseline


def save_telemetry_calibration(data: Dict[str, Any]) -> None:
    TELEMETRY_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["weights"] = _normalize_telemetry_group_weights(payload.get("weights", {}))
    payload.setdefault("reference", dict(DEFAULT_TELEMETRY_REFERENCE))
    payload.setdefault("scales", dict(DEFAULT_TELEMETRY_SCALES))
    payload.setdefault("idle_floor", 0.14)
    payload.setdefault("warmup_samples", 6)
    TELEMETRY_CALIBRATION_PATH.write_text(safe_json_dumps(payload), encoding="utf-8")


def logistic(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def mean(values: Sequence[float]) -> float:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    values = [float(v) for v in values if v is not None]
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def median(values: Sequence[float]) -> float:
    values = [float(v) for v in values if v is not None]
    return float(statistics.median(values)) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return 0.0
    if p <= 0:
        return values[0]
    if p >= 1:
        return values[-1]
    idx = (len(values) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def minmax_scale(values: Sequence[float]) -> List[float]:
    values = [float(v) for v in values]
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if math.isclose(mn, mx):
        return [0.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


def pbkdf2_hash(password: str, salt: Optional[bytes] = None) -> str:
    if not password:
        raise ValueError("password is empty")
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return base64.b64encode(salt + digest).decode("ascii")


def pbkdf2_verify(password: str, encoded: str) -> bool:
    try:
        raw = base64.b64decode(encoded.encode("ascii"))
        salt, digest = raw[:16], raw[16:]
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
        return hmac.compare_digest(candidate, digest)
    except Exception:
        return False


def role_label(role: str) -> str:
    return {"player": "Игрок", "researcher": "Исследователь"}.get(role, role)


def profile_from_group_choice(full_name: str, age: Any, sex: str, disease_group: Any, notes: str = "") -> Dict[str, Any]:
    group = normalize_disease_group(disease_group)
    return {
        "full_name": (full_name or "").strip(),
        "age": int(to_float(age, 0)),
        "sex": (sex or "").strip(),
        "healthy": group == "healthy",
        "asthma": group == "asthma",
        "lung": group == "asthma",
        "heart": group == "heart",
        "cardio": group == "heart",
        "other": False,
        "disease_group": group,
        "notes": (notes or "").strip(),
    }


def disease_code(category: str, number: int) -> str:
    prefix = DISEASE_CODES.get(category, "О")
    return f"{prefix}{number}"


def normalize_disease_group(value: Any) -> str:
    value = str(value or "healthy").strip().lower()
    aliases = {
        "здоров": "healthy",
        "здоровый": "healthy",
        "здоровые": "healthy",
        "з": "healthy",
        "healthy": "healthy",
        "астма": "asthma",
        "лёгкие": "asthma",
        "легкие": "asthma",
        "lungs": "asthma",
        "lung": "asthma",
        "asthma": "asthma",
        "сердце": "heart",
        "сердечные": "heart",
        "cardio": "heart",
        "heart": "heart",
    }
    return aliases.get(value, value if value in DISEASE_GROUP_CHOICES else "healthy")


def classify_disease_group(profile: Dict[str, Any]) -> str:
    """Возвращает одну из трёх групп сравнения."""
    raw_group = profile.get("disease_group")
    if raw_group is not None and str(raw_group).strip():
        group = normalize_disease_group(raw_group)
        if group in DISEASE_GROUP_CHOICES:
            return group
    if profile.get("healthy"):
        return "healthy"
    flags = {k for k, v in profile.items() if bool(v)}
    if flags & {"asthma", "lung", "respiratory"}:
        return "asthma"
    if flags & {"heart", "cardio", "arrhythmia", "hypertension"}:
        return "heart"
    return "healthy"


def disease_codes_for_profile(profile: Dict[str, Any], number: int = 1) -> str:
    group = classify_disease_group(profile)
    return disease_code(group, number)


def validate_username(username: str) -> str:
    username = (username or "").strip()
    if not username:
        raise ValueError("Имя пользователя не может быть пустым")
    if len(username) < 3:
        raise ValueError("Имя пользователя должно содержать минимум 3 символа")
    return username


def validate_password(password: str) -> str:
    password = password or ""
    if not password.strip():
        raise ValueError("Пароль не может быть пустым")
    if len(password) < 4:
        raise ValueError("Пароль должен содержать минимум 4 символа")
    return password


def rr_from_hr_series(hr_series: Sequence[float]) -> List[float]:
    """Грубая оценка RR в миллисекундах из ЧСС, если отдельные RR не доступны."""
    out = []
    for hr in hr_series:
        hr = float(hr)
        if hr > 0:
            out.append(60_000.0 / hr)
    return out


def sdnn(rr_ms: Sequence[float]) -> float:
    rr = [float(v) for v in rr_ms if v is not None]
    return stdev(rr)


def rmssd(rr_ms: Sequence[float]) -> float:
    rr = [float(v) for v in rr_ms if v is not None]
    if len(rr) < 2:
        return 0.0
    diffs = [(rr[i] - rr[i - 1]) ** 2 for i in range(1, len(rr))]
    return math.sqrt(sum(diffs) / len(diffs)) if diffs else 0.0


def pnn50(rr_ms: Sequence[float]) -> float:
    rr = [float(v) for v in rr_ms if v is not None]
    if len(rr) < 2:
        return 0.0
    diffs = [abs(rr[i] - rr[i - 1]) for i in range(1, len(rr))]
    return sum(1 for d in diffs if d > 50.0) / len(diffs)


def baevsky_stress_index(rr_ms: Sequence[float]) -> float:
    """Индекс напряжения по Баевскому: AMo*100/(2*Mo*MxDMn)."""
    rr = [float(v) for v in rr_ms if v is not None and v > 0]
    if len(rr) < 3:
        return 0.0
    mn, mx = min(rr), max(rr)
    mxdmn = max(mx - mn, 1e-6)
    # Бинирование 50 мс соответствует практическому построению гистограммы RR
    bins: Dict[int, int] = {}
    for val in rr:
        bin_id = int(round(val / 50.0) * 50)
        bins[bin_id] = bins.get(bin_id, 0) + 1
    mode_rr, mode_count = max(bins.items(), key=lambda kv: kv[1])
    amo = 100.0 * mode_count / len(rr)
    mo = float(mode_rr)
    if mo <= 0.0:
        return 0.0
    return (amo * 100.0) / (2.0 * mo * mxdmn)


def hr_summary(hr_series: Sequence[float], rr_series: Optional[Sequence[float]] = None) -> Dict[str, float]:
    hr = [float(v) for v in hr_series if v is not None]
    rr = [float(v) for v in (rr_series or []) if v is not None]
    return {
        "hr_mean": mean(hr),
        "hr_std": stdev(hr),
        "rr_mean": mean(rr),
        "sdnn": sdnn(rr) if rr else 0.0,
        "rmssd": rmssd(rr) if rr else 0.0,
        "pnn50": pnn50(rr) if rr else 0.0,
        "baevsky": baevsky_stress_index(rr) if rr else 0.0,
    }


def telemetry_summary(events: Sequence[Dict[str, Any]], window_seconds: float = 10.0) -> Dict[str, float]:
    """Считает поведенческие признаки из телеметрии.

    Для телеметрии мыши используются сгруппированные сегменты движения
    (скорость, ускорение, jerk, смены направления, эффективность траектории),
    а не сырые микрособытия по каждому пикселю. Это делает метрики устойчивее
    к обычной плавной игре и снижает ложные пики на старте сессии.
    """
    base = {
        "click_rate": 0.0,
        "error_rate": 0.0,
        "mouse_speed_mean": 0.0,
        "mouse_speed_std": 0.0,
        "mouse_speed_max": 0.0,
        "accel_mean": 0.0,
        "accel_max": 0.0,
        "jerk_mean": 0.0,
        "key_rate": 0.0,
        "key_hold_mean": 0.0,
        "key_hold_std": 0.0,
        "key_hold_max": 0.0,
        "event_density": 0.0,
        "path_length": 0.0,
        "path_efficiency": 0.0,
        "motion_duration": 0.0,
        "motion_segments": 0.0,
        "motion_samples": 0.0,
        "motion_turns": 0.0,
        "motion_turn_rate": 0.0,
        "motion_speed_p90": 0.0,
        "motion_accel_p90": 0.0,
        "motion_jerk_p90": 0.0,
        "motion_straightness": 0.0,
        "double_click_rate": 0.0,
        "pause_count": 0.0,
        "backspace_count": 0.0,
        "escape_count": 0.0,
        "scroll_count": 0.0,
        "mouse_reversal_rate": 0.0,
        "mouse_spike_rate": 0.0,
        "focus_loss_rate": 0.0,
        "mouse_sample_count": 0.0,
        "mouse_move_rate": 0.0,
    }
    if not events:
        return base

    total = max(window_seconds, 1e-6)
    mouse = [e for e in events if e.get("type") in {"mouse_move", "motion_segment"}]
    clicks = [e for e in events if e.get("type") == "click"]
    keys = [e for e in events if e.get("type") == "key_down"]
    ups = [e for e in events if e.get("type") == "key_up"]
    backspaces = [e for e in events if e.get("type") in {"backspace", "correction"}]
    escapes = [e for e in events if e.get("type") in {"escape", "focus_loss"}]
    scrolls = [e for e in events if e.get("type") == "scroll"]
    spikes = [e for e in events if e.get("type") in {"mouse_spike", "scroll_burst"}]

    speeds: List[float] = []
    accels: List[float] = []
    jerks: List[float] = []
    straightness_values: List[float] = []
    path_length = 0.0
    motion_duration = 0.0
    motion_turns = 0.0
    motion_samples = 0.0
    motion_segments = 0

    click_times = [to_float(e.get("t", 0.0)) for e in clicks]
    double_click_count = sum(1 for i in range(1, len(click_times)) if click_times[i] - click_times[i - 1] <= 0.25)
    key_holds = [to_float(e.get("hold", 0.0)) for e in ups if to_float(e.get("hold", 0.0)) > 0]

    pauses = 0
    reversals = 0
    last_ts = None
    for e in mouse:
        ts = to_float(e.get("t", 0.0))
        duration = max(to_float(e.get("duration", 0.0)), 0.0)
        path = max(to_float(e.get("path", e.get("dist", 0.0))), 0.0)
        net_dist = max(to_float(e.get("net_dist", math.hypot(to_float(e.get("dx", 0.0)), to_float(e.get("dy", 0.0))))), 0.0)
        speed = to_float(e.get("speed", path / max(duration, 1e-3) if duration > 0 else 0.0))
        accel = to_float(e.get("accel", 0.0))
        jerk = to_float(e.get("jerk", 0.0))
        turns = to_float(e.get("turns", 0.0))
        samples = to_float(e.get("samples", 1.0))
        straightness = to_float(e.get("straightness", net_dist / path if path > 0 else 1.0))
        pause_after = to_float(e.get("pause_after", 0.0))
        spike_flag = 1 if e.get("type") == "mouse_spike" else 0
        if last_ts is not None and ts - last_ts >= 0.8:
            pauses += 1
        pauses += 1 if pause_after >= 0.35 else 0
        reversals += int(turns > 0)
        motion_duration += duration if duration > 0 else 0.03
        path_length += path if path > 0 else 0.0
        motion_turns += max(0.0, turns)
        motion_samples += max(1.0, samples)
        motion_segments += 1
        if path > 0:
            speeds.append(speed)
            accels.append(accel)
            jerks.append(jerk)
            straightness_values.append(clip01(straightness))
        if spike_flag:
            spikes.append(e)
        last_ts = ts

    active = max(motion_duration, 1e-6)
    path_efficiency = (sum(to_float(e.get("net_dist", 0.0)) for e in mouse) / path_length) if path_length > 0 else 1.0
    if not 0.0 <= path_efficiency <= 1.0:
        path_efficiency = clip01(path_efficiency)

    base.update({
        "click_rate": len(clicks) / total,
        "error_rate": (len(backspaces) + len(escapes)) / total,
        "mouse_speed_mean": median(speeds),
        "mouse_speed_std": stdev(speeds),
        "mouse_speed_max": percentile(speeds, 0.90) if speeds else 0.0,
        "accel_mean": median(accels),
        "accel_max": percentile(accels, 0.90) if accels else 0.0,
        "jerk_mean": median(jerks),
        "key_rate": len(keys) / total,
        "key_hold_mean": mean(key_holds),
        "key_hold_std": stdev(key_holds),
        "key_hold_max": max(key_holds) if key_holds else 0.0,
        "event_density": len(events) / total,
        "path_length": path_length,
        "path_efficiency": clip01(path_efficiency),
        "motion_duration": motion_duration,
        "motion_segments": float(motion_segments),
        "motion_samples": float(motion_samples),
        "motion_turns": float(motion_turns),
        "motion_turn_rate": motion_turns / active,
        "motion_speed_p90": percentile(speeds, 0.90) if speeds else 0.0,
        "motion_accel_p90": percentile(accels, 0.90) if accels else 0.0,
        "motion_jerk_p90": percentile(jerks, 0.90) if jerks else 0.0,
        "motion_straightness": mean(straightness_values) if straightness_values else clip01(path_efficiency),
        "double_click_rate": double_click_count / total,
        "pause_count": pauses / total,
        "backspace_count": len(backspaces) / total,
        "escape_count": len(escapes) / total,
        "scroll_count": len(scrolls) / total,
        "mouse_reversal_rate": reversals / max(motion_segments, 1),
        "mouse_spike_rate": len(spikes) / total,
        "focus_loss_rate": len(escapes) / total,
        "mouse_sample_count": float(motion_segments),
        "mouse_move_rate": motion_segments / total,
    })
    return base


def _feature_level(value: float, reference: float, scale: float, invert: bool = False) -> float:
    reference = float(reference)
    scale = max(float(scale), 1e-6)
    value = float(value)
    if invert:
        return logistic((reference - value) / scale)
    return logistic((value - reference) / scale)


def telemetry_stress_score(feat: Dict[str, float]) -> float:
    """Поведенческий индекс стресса на основе HCI-исследований.

    Модель теперь использует калибруемые группы признаков и динамическое
    приглушение первых секунд сессии. Это снижает ложные пики на старте и
    делает показатель ближе к результатам работ по mouse dynamics, где
    stress-эффекты проявляются через speed-accuracy trade-off и смещение
    распределений скоростей/пауз, а не через единичный рывок мыши.
    """
    calib = load_telemetry_calibration()
    reference = calib.get("reference", DEFAULT_TELEMETRY_REFERENCE)
    scales = calib.get("scales", DEFAULT_TELEMETRY_SCALES)
    weights = _normalize_telemetry_group_weights(calib.get("weights", DEFAULT_TELEMETRY_WEIGHTS))
    idle_floor = clip01(to_float(calib.get("idle_floor", 0.14), 0.14))
    warmup_samples = max(3.0, to_float(calib.get("warmup_samples", 6), 6))

    sample_count = max(0.0, feat.get("mouse_sample_count", 0.0))
    if sample_count <= 0.0:
        return idle_floor * 0.5

    # Группа моторного напряжения: скорость, ускорение, jerk и развороты.
    motor_terms = [
        _feature_level(feat.get("mouse_speed_mean", 0.0), reference.get("mouse_speed_mean", 0.0), scales.get("mouse_speed_mean", 1.0)),
        _feature_level(feat.get("mouse_speed_std", 0.0), reference.get("mouse_speed_std", 0.0), scales.get("mouse_speed_std", 1.0)),
        _feature_level(feat.get("mouse_speed_max", 0.0), reference.get("mouse_speed_max", 0.0), scales.get("mouse_speed_max", 1.0)),
        _feature_level(feat.get("accel_mean", 0.0), reference.get("accel_mean", 0.0), scales.get("accel_mean", 1.0)),
        _feature_level(feat.get("accel_max", 0.0), reference.get("accel_max", 0.0), scales.get("accel_max", 1.0)),
        _feature_level(feat.get("jerk_mean", 0.0), reference.get("jerk_mean", 0.0), scales.get("jerk_mean", 1.0)),
        _feature_level(feat.get("motion_turn_rate", 0.0), reference.get("motion_turn_rate", 0.0), scales.get("motion_turn_rate", 1.0)),
        _feature_level(feat.get("mouse_reversal_rate", 0.0), reference.get("mouse_reversal_rate", 0.0), scales.get("mouse_reversal_rate", 1.0)),
        _feature_level(feat.get("mouse_spike_rate", 0.0), reference.get("mouse_spike_rate", 0.0), scales.get("mouse_spike_rate", 1.0)),
    ]
    motor = mean(motor_terms)

    # Контроль траектории: чем меньше эффективность/прямолинейность, тем выше напряжение.
    control_terms = [
        _feature_level(feat.get("path_efficiency", 0.0), reference.get("path_efficiency", 0.0), scales.get("path_efficiency", 1.0), invert=True),
        _feature_level(feat.get("motion_straightness", 0.0), reference.get("motion_straightness", 0.0), scales.get("motion_straightness", 1.0), invert=True),
        _feature_level(feat.get("mouse_move_rate", 0.0), reference.get("mouse_move_rate", 0.0), scales.get("mouse_move_rate", 1.0)),
    ]
    control = mean(control_terms)

    # Когнитивное напряжение: паузы и нестабильность удержания клавиш.
    cognitive_terms = [
        _feature_level(feat.get("pause_count", 0.0), reference.get("pause_count", 0.0), scales.get("pause_count", 1.0)),
        _feature_level(feat.get("key_hold_std", 0.0), reference.get("key_hold_std", 0.0), scales.get("key_hold_std", 1.0)),
        _feature_level(feat.get("key_rate", 0.0), 0.55, 0.35),
    ]
    cognitive = mean(cognitive_terms)

    # Поведенческое напряжение: ошибки, коррекции и потеря фокуса.
    behavioral_terms = [
        _feature_level(feat.get("error_rate", 0.0), reference.get("error_rate", 0.0), scales.get("error_rate", 1.0)),
        _feature_level(feat.get("backspace_count", 0.0), reference.get("backspace_count", 0.0), scales.get("backspace_count", 1.0)),
        _feature_level(feat.get("escape_count", 0.0), reference.get("escape_count", 0.0), scales.get("escape_count", 1.0)),
        _feature_level(feat.get("scroll_count", 0.0), reference.get("scroll_count", 0.0), scales.get("scroll_count", 1.0)),
        _feature_level(feat.get("double_click_rate", 0.0), reference.get("double_click_rate", 0.0), scales.get("double_click_rate", 1.0)),
        _feature_level(feat.get("focus_loss_rate", 0.0), reference.get("escape_count", 0.0), scales.get("escape_count", 1.0)),
    ]
    behavioral = mean(behavioral_terms)

    raw = (
        weights.get("motor", DEFAULT_TELEMETRY_WEIGHTS["motor"]) * motor
        + weights.get("control", DEFAULT_TELEMETRY_WEIGHTS["control"]) * control
        + weights.get("cognitive", DEFAULT_TELEMETRY_WEIGHTS["cognitive"]) * cognitive
        + weights.get("behavioral", DEFAULT_TELEMETRY_WEIGHTS["behavioral"]) * behavioral
    )

    # Дросселирование старта: до накопления достаточного числа сегментов
    # индекс не должен скакать в высокую область из-за единичного рывка.
    confidence = clip01((sample_count - warmup_samples) / max(warmup_samples * 2.0, 1e-6))
    if sample_count < warmup_samples:
        return clip01(idle_floor * (0.65 + 0.35 * confidence))

    score = idle_floor * (1.0 - confidence) + raw * confidence
    if sample_count < warmup_samples * 2.0:
        score = min(score, 0.40)
    return clip01(score)


def heart_stress_score(feat: Dict[str, float]) -> float:
    """Нормированный физиологический подиндекс стресса."""
    hr_mean = feat.get("hr_mean", 0.0)
    hr_std = feat.get("hr_std", 0.0)
    rmssd_value = feat.get("rmssd", 0.0)
    sdnn_value = feat.get("sdnn", 0.0)
    baevsky = feat.get("baevsky", 0.0)
    pnn50_value = feat.get("pnn50", 0.0)

    score = (
        0.34 * logistic((hr_mean - 78.0) / 10.0)
        + 0.14 * logistic((hr_std - 8.0) / 4.0)
        + 0.18 * (1.0 - logistic((rmssd_value - 35.0) / 12.0))
        + 0.12 * (1.0 - logistic((sdnn_value - 50.0) / 15.0))
        + 0.14 * logistic((baevsky - 100.0) / 60.0)
        + 0.08 * (1.0 - logistic((pnn50_value - 0.20) / 0.08))
    )
    return clip01(score)


@dataclass
class StressParts:
    heart_score: float
    telemetry_score: float
    ml_score: float
    overall_score: float
    source_notes: str = ""


def combine_stress_scores(
    heart: Optional[float],
    telemetry: Optional[float],
    ml: Optional[float] = 0.5,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Сводит подиндексы в итоговый балл.

    Если weights не переданы, используются калиброванные веса из
    rule_calibration.json. Когда калибровка отсутствует, берется
    априорный набор весов DEFAULT_WEIGHTS.
    """
    raw_weights = load_rule_calibration().get("weights", DEFAULT_WEIGHTS) if weights is None else weights
    weights = _normalize_weight_map(dict(raw_weights))
    parts = {"heart": heart, "telemetry": telemetry, "ml": ml}
    active = {k: float(v) for k, v in parts.items() if v is not None}
    if not active:
        return 0.0
    norm = sum(weights.get(k, 0.0) for k in active) or 1.0
    total = sum(weights.get(k, 0.0) / norm * clip01(float(v)) for k, v in active.items())
    return clip01(total)


def stress_class(score: float) -> str:
    if score < 0.35:
        return "низкий"
    if score < 0.70:
        return "средний"
    return "высокий"


def percent_diff(a: float, b: float) -> float:
    a = float(a)
    b = float(b)
    if math.isclose(b, 0.0):
        return 0.0
    return (a - b) / abs(b) * 100.0


def safe_json_loads(text: str, default: Any = None) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def safe_json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def choose_label_from_prob(prob: float) -> str:
    return "high" if prob >= 0.7 else "medium" if prob >= 0.35 else "low"


def _feature_context(feature_row: Dict[str, Any]) -> Dict[str, Any]:
    """Собирает плоский словарь признаков из возможных вложенных структур.

    Для исторических записей признаки могут лежать в payload.summary.metrics,
    а для онлайн-предсказания приходят сразу в плоском виде. Эта функция
    объединяет оба варианта, чтобы ML-модель обучалась и предсказывала на
    одном и том же наборе признаков.
    """
    if not isinstance(feature_row, dict):
        return {}

    context: Dict[str, Any] = {}
    context.update(feature_row)

    for key in ("summary", "features", "feature_snapshot", "metrics"):
        nested = feature_row.get(key)
        if isinstance(nested, dict):
            context.update(nested)
            for subkey in ("metrics", "features", "feature_snapshot"):
                subnested = nested.get(subkey)
                if isinstance(subnested, dict):
                    context.update(subnested)
    return context


def feature_vector_for_model(feature_row: Dict[str, Any]) -> List[float]:
    # Модель обучается и работает на одном и том же наборе признаков.
    data = _feature_context(feature_row)

    keys = [
        "hr_mean", "hr_std", "sdnn", "rmssd", "pnn50", "baevsky",
        "click_rate", "error_rate", "mouse_speed_mean", "mouse_speed_std",
        "mouse_speed_max", "accel_mean", "accel_max", "jerk_mean",
        "event_density", "path_length", "path_efficiency", "motion_duration",
        "motion_segments", "motion_turn_rate", "motion_speed_p90",
        "motion_accel_p90", "motion_jerk_p90", "motion_straightness",
        "double_click_rate", "pause_count", "backspace_count", "escape_count",
        "scroll_count", "mouse_reversal_rate", "mouse_spike_rate",
        "focus_loss_rate", "mouse_sample_count", "mouse_move_rate",
        "key_rate", "key_hold_mean", "key_hold_std", "key_hold_max",
    ]
    return [to_float(data.get(k, 0.0)) for k in keys]


def feature_names_for_model() -> List[str]:
    return [
        "hr_mean", "hr_std", "sdnn", "rmssd", "pnn50", "baevsky",
        "click_rate", "error_rate", "mouse_speed_mean", "mouse_speed_std",
        "mouse_speed_max", "accel_mean", "accel_max", "jerk_mean",
        "event_density", "path_length", "path_efficiency", "motion_duration",
        "motion_segments", "motion_turn_rate", "motion_speed_p90",
        "motion_accel_p90", "motion_jerk_p90", "motion_straightness",
        "double_click_rate", "pause_count", "backspace_count", "escape_count",
        "scroll_count", "mouse_reversal_rate", "mouse_spike_rate",
        "focus_loss_rate", "mouse_sample_count", "mouse_move_rate",
        "key_rate", "key_hold_mean", "key_hold_std", "key_hold_max",
    ]


class StressML:
    """Модель ML для вероятности высокого стресса + калибровка подиндексов."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else MODELS_DIR / "stress_ml.joblib"
        self.model: Optional[Pipeline] = None
        self.combiner: Optional[LogisticRegression] = None
        self.load()

    def load(self) -> None:
        if self.model_path.exists():
            try:
                artifact = joblib.load(self.model_path)
                self.model = artifact.get("model")
                self.combiner = artifact.get("combiner")
            except Exception:
                self.model = None
                self.combiner = None

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "combiner": self.combiner}, self.model_path)

    def fit(self, rows: Sequence[Dict[str, Any]], labels: Sequence[int]) -> None:
        X = [feature_vector_for_model(r) for r in rows]
        y = [int(v) for v in labels]
        if not X or len(set(y)) < 2:
            return

        from sklearn.model_selection import GridSearchCV

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=42, class_weight="balanced"))
        ])

        # Если данных мало, учим базово. Иначе — ищем оптимальные гиперпараметры.
        if len(X) < 15:
            self.model = pipe.set_params(rf__n_estimators=100, rf__max_depth=5)
            self.model.fit(X, y)
        else:
            param_grid = {
                'rf__n_estimators': [80, 120, 180],
                'rf__max_depth': [4, 6, 8, None]
            }
            # Используем кросс-валидацию, подстраиваясь под малое количество данных
            cv_folds = min(3, np.bincount(y).min())
            if cv_folds < 2:
                self.model = pipe.set_params(rf__n_estimators=100)
                self.model.fit(X, y)
            else:
                grid = GridSearchCV(pipe, param_grid, cv=cv_folds, scoring='f1_macro')
                grid.fit(X, y)
                self.model = grid.best_estimator_

        # Калибратор для итогового индекса: отдельная логистическая регрессия
        # по трём подиндексам. Здесь не используем несуществующий параметр
        # positive=True, а после обучения работаем через predict_proba.
        subX = np.asarray([[to_float(r.get("heart_score", 0.0)),
                            to_float(r.get("telemetry_score", 0.0)),
                            to_float(r.get("ml_prob", 0.5))] for r in rows], dtype=float)
        if len(set(y)) >= 2 and len(subX) >= 2:
            self.combiner = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ])
            try:
                self.combiner.fit(subX, y)
            except Exception:
                self.combiner = None
        self.save()

    def _prior_prob(self, row: Dict[str, Any]) -> float:
        ctx = _feature_context(row)
        weights = load_rule_calibration().get("weights", DEFAULT_WEIGHTS)
        heart = heart_stress_score(ctx) if any(to_float(ctx.get(k, 0.0)) for k in ("hr_mean", "hr_std", "sdnn", "rmssd", "pnn50", "baevsky")) else None
        telemetry = telemetry_stress_score(ctx)
        prior = combine_stress_scores(heart, telemetry, 0.5, weights=weights)
        sample_count = to_float(ctx.get("mouse_sample_count", 0.0), 0.0)
        if sample_count < 4:
            prior = min(prior, 0.32)
        return clip01(prior)

    def predict_prob(self, row: Dict[str, Any]) -> float:
        prior = self._prior_prob(row)
        if self.model is None:
            return prior
        X = np.asarray([feature_vector_for_model(row)], dtype=float)
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)[0]
                model_prob = float(proba[-1]) if len(proba) > 1 else float(proba[0])
            else:
                model_prob = float(self.model.predict(X)[0])
        except Exception:
            return prior
        # Модель ML не должна быть изолированной от игровой динамики: если
        # она дает почти постоянный ответ, prior от поведенческих признаков
        # удерживает выход от деградации в одну фиксированную величину.
        blended = 0.68 * model_prob + 0.32 * prior
        return clip01(blended)

    def combine(self, heart_score: float, telemetry_score: float, ml_prob: float) -> float:
        if self.combiner is None:
            return combine_stress_scores(heart_score, telemetry_score, ml_prob)
        try:
            x = np.asarray([[heart_score, telemetry_score, ml_prob]], dtype=float)
            if hasattr(self.combiner, "predict_proba"):
                return float(self.combiner.predict_proba(x)[0, 1])
            return float(self.combiner.predict(x)[0])
        except Exception:
            return combine_stress_scores(heart_score, telemetry_score, ml_prob)

            return combine_stress_scores(heart_score, telemetry_score, ml_prob)
        try:
            x = np.asarray([[heart_score, telemetry_score, ml_prob]], dtype=float)
            if hasattr(self.combiner, "predict_proba"):
                return float(self.combiner.predict_proba(x)[0, 1])
            return float(self.combiner.predict(x)[0])
        except Exception:
            return combine_stress_scores(heart_score, telemetry_score, ml_prob)


def calibrate_rule_weights(rows: Sequence[Dict[str, Any]], labels: Sequence[int]) -> Dict[str, float]:
    """Оценивает веса rule-based модели по пилотным данным.

    Вместо фиксированного ручного набора коэффициентов используется
    обучаемая логистическая регрессия на подиндексах. Модуль возвращает
    нормированные положительные веса, которые затем сохраняются в
    rule_calibration.json и используются при расчете итогового индекса.
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return dict(DEFAULT_WEIGHTS)

    X = np.asarray([[
        to_float(r.get("heart_score", 0.0)),
        to_float(r.get("telemetry_score", 0.0)),
        to_float(r.get("ml_score", 0.0)),
    ] for r in rows], dtype=float)
    y = np.asarray([int(v) for v in labels], dtype=int)

    if len(X) < 6 or len(set(y.tolist())) < 2:
        return dict(DEFAULT_WEIGHTS)

    # Оцениваем связь между подиндексами и целевой меткой через
    # регуляризованную логистическую регрессию. Это делает вклад каждого
    # канала обоснованным данными, а не жестко заданным.
    try:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ])
        model.fit(X, y)
        coef = np.abs(model.named_steps["lr"].coef_[0])
    except Exception:
        return dict(DEFAULT_WEIGHTS)

    if not np.isfinite(coef).all() or float(np.sum(coef)) <= 1e-12:
        return dict(DEFAULT_WEIGHTS)

    # Чем сильнее коэффициент, тем больше итоговый вес. Нормируем в сумму 1.
    coef = coef / float(np.sum(coef))
    return {
        "heart": float(coef[0]),
        "telemetry": float(coef[1]),
        "ml": float(coef[2]),
    }


def _group_median(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if v is not None]
    return float(statistics.median(vals)) if vals else float(default)


def calibrate_telemetry_model(rows: Sequence[Dict[str, Any]], labels: Sequence[int]) -> Dict[str, Any]:
    """Строит калибровку поведенческой телеметрии по пилотным данным.

    Веса и пороги выводятся из разницы между низким и высоким стрессом на
    наблюдаемых сессиях. Это делает телеметрию более согласованной с
    данными исследования и уменьшает зависимость от жестко заданных порогов.
    """
    groups = {
        "motor": ["mouse_speed_mean", "mouse_speed_std", "mouse_speed_max", "accel_mean", "accel_max", "jerk_mean", "motion_turn_rate", "mouse_reversal_rate", "mouse_spike_rate"],
        "control": ["path_efficiency", "motion_straightness", "mouse_move_rate"],
        "cognitive": ["pause_count", "key_hold_std", "key_rate"],
        "behavioral": ["error_rate", "backspace_count", "escape_count", "scroll_count", "double_click_rate", "focus_loss_rate"],
    }
    if not rows or len(rows) < 4 or len(set(int(v) for v in labels)) < 2:
        return {
            "weights": dict(DEFAULT_TELEMETRY_WEIGHTS),
            "reference": dict(DEFAULT_TELEMETRY_REFERENCE),
            "scales": dict(DEFAULT_TELEMETRY_SCALES),
            "idle_floor": 0.14,
            "warmup_samples": 6,
            "threshold": 0.5,
            "calibrated": False,
            "source_rows": len(rows),
            "updated_at": now_iso(),
        }

    low_rows = [r for r, y in zip(rows, labels) if int(y) == 0]
    high_rows = [r for r, y in zip(rows, labels) if int(y) == 1]

    reference: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    effect_by_group: Dict[str, float] = {}
    for group, keys in groups.items():
        effects: List[float] = []
        for key in keys:
            low_vals = [to_float(_feature_context(r).get(key, 0.0)) for r in low_rows]
            high_vals = [to_float(_feature_context(r).get(key, 0.0)) for r in high_rows]
            all_vals = low_vals + high_vals
            ref_val = _group_median(low_vals, _group_median(all_vals, DEFAULT_TELEMETRY_REFERENCE.get(key, 0.0)))
            high_val = _group_median(high_vals, ref_val)
            spread = percentile(all_vals, 0.75) - percentile(all_vals, 0.25) if len(all_vals) >= 4 else 0.0
            scale = max(abs(high_val - ref_val), abs(spread) / 1.349 if spread else 0.0, DEFAULT_TELEMETRY_SCALES.get(key, 1.0), 1e-6)
            reference[key] = float(ref_val)
            scales[key] = float(scale)
            effects.append(abs(high_val - ref_val) / scale if scale > 0 else 0.0)
        effect_by_group[group] = mean(effects) if effects else DEFAULT_TELEMETRY_WEIGHTS[group]

    weights = _normalize_telemetry_group_weights(effect_by_group)
    # Чем больше effect size между классами, тем ниже порог для роста индекса.
    group_strength = mean(list(weights.values())) if weights else 0.25
    threshold = clip01(0.45 + (0.10 * (0.25 - group_strength)))
    return {
        "weights": weights,
        "reference": reference or dict(DEFAULT_TELEMETRY_REFERENCE),
        "scales": scales or dict(DEFAULT_TELEMETRY_SCALES),
        "idle_floor": 0.14,
        "warmup_samples": 6,
        "threshold": threshold,
        "calibrated": True,
        "source_rows": len(rows),
        "updated_at": now_iso(),
    }


def calibrate_rule_model(rows: Sequence[Dict[str, Any]], labels: Sequence[int]) -> Dict[str, Any]:
    """Строит калибровку rule-based модели: веса + порог."""
    weights = calibrate_rule_weights(rows, labels)
    scores = [combine_stress_scores(r.get("heart_score"), r.get("telemetry_score"), r.get("ml_score"), weights=weights) for r in rows]
    threshold = _threshold_from_scores(labels, scores)
    return {
        "weights": weights,
        "threshold": threshold,
        "calibrated": True,
        "source_rows": len(rows),
        "updated_at": now_iso(),
    }


def payload_from_group_choice(full_name: str, age: Any, sex: str, disease_group: Any, notes: str = "") -> Dict[str, Any]:
    return profile_from_group_choice(full_name, age, sex, disease_group, notes)


def active_window_title() -> str:
    try:
        import pygetwindow as gw  # type: ignore
        win = gw.getActiveWindow()
        if win and getattr(win, "title", None):
            return str(win.title).strip()
    except Exception:
        pass
    return ""


def looks_like_game_window(title: str, app_name: str = APP_NAME) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return False
    bad = (app_name.lower(), "python", "explorer", "file explorer", "desktop", "browser", "visual studio", "code", "terminal", "cmd", "powershell")
    return not any(b in t for b in bad)


def wait_for_game_window(timeout: float = 0.0, hint: str = "") -> str:
    start = time.time()
    hint = (hint or "").strip().lower()
    while True:
        title = active_window_title()
        if title and looks_like_game_window(title):
            if not hint or hint in title.lower():
                return title
        if timeout and (time.time() - start) >= timeout:
            return title
        time.sleep(0.25)



def sample_summary_text(parts: StressParts, has_rr: bool) -> str:
    rr_text = "RR-данные доступны" if has_rr else "RR-данные отсутствуют, HRV-метрики отключены"
    return (
        f"ЧСС/HRV: {parts.heart_score:.2f}; "
        f"Телеметрия: {parts.telemetry_score:.2f}; "
        f"ML: {parts.ml_score:.2f}; "
        f"Итог: {parts.overall_score:.2f}. {rr_text}"
    )
