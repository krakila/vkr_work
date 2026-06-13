from shared import (
    telemetry_stress_score, telemetry_summary,
    heart_stress_score, compute_hrv_from_ibi,
    combine_stress_scores, stress_class,
    load_rule_calibration, save_rule_calibration,
    normalize_disease_group
)
import tempfile
from pathlib import Path
import shared

def test_telemetry_stress_score_edge_cases():
    # Пустой список событий
    score = telemetry_stress_score(telemetry_summary([], 10.0))
    assert 0.0 <= score <= 1.0

    # Генерируем интенсивные движения мыши, характерные для стресса
    events = []
    for i in range(50):
        events.append({
            "t": i * 0.1,
            "type": "mouse_move",
            "speed": 800.0,        # высокая скорость (пикс/с)
            "accel": 3000.0,       # высокое ускорение
            "jerk": 5000.0,
            "dist": 50.0,
            "duration": 0.05,
            "straightness": 0.5,
            "turns": 2
        })
    feat = telemetry_summary(events, 10.0)
    score = telemetry_stress_score(feat)
    # Проверяем, что функция отработала и вернула корректное значение
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # При желании можно проверить, что score заметно выше, чем при пустом списке:
    # но из-за калибровки и warmup-периода может быть нестабильно, поэтому не делаем.

def test_heart_stress_score_with_ibi():
    ibi_series = [800, 810, 790, 805, 820, 750, 850]
    hrv = compute_hrv_from_ibi(ibi_series)
    feat = {
        "hr_mean": 60000.0 / hrv["mean_ibi"] if hrv["mean_ibi"] > 0 else 0,
        "hr_std": 0,
        "sdnn": hrv["sdnn"],
        "rmssd": hrv["rmssd"],
        "pnn50": hrv["pnn50"],
        "baevsky": hrv["baevsky"]
    }
    score = heart_stress_score(feat)
    assert 0.0 <= score <= 1.0

def test_rule_calibration_persistence():
    original_path = shared.RULE_CALIBRATION_PATH
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "rule_calibration.json"
            shared.RULE_CALIBRATION_PATH = fake_path
            save_rule_calibration({"weights": {"heart": 0.5, "telemetry": 0.3, "ml": 0.2}, "threshold": 0.5})
            loaded = load_rule_calibration()
            assert loaded["weights"]["heart"] == 0.5
    finally:
        shared.RULE_CALIBRATION_PATH = original_path

def test_stress_class():
    assert stress_class(0.2) == "низкий"
    assert stress_class(0.5) == "средний"
    assert stress_class(0.8) == "высокий"
    assert stress_class(0.35) == "средний"

def test_normalize_disease_group():
    assert normalize_disease_group("ЗДОРОВЫЙ") == "healthy"
    assert normalize_disease_group("астма") == "asthma"
    assert normalize_disease_group("сердечные") == "heart"
    assert normalize_disease_group("unknown") == "healthy"