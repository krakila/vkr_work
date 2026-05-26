
"""REST API для клиент-серверной ИС анализа стресса."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import jwt
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from shared import (
    APP_NAME,
    CACHE_DIR,
    DATA_DIR,
    DB_NAME,
    MEDIA_DIR,
    MODELS_DIR,
    ROLE_PLAYER,
    ROLE_RESEARCHER,
    ROLES,
    DEFAULT_WEIGHTS,
    load_rule_calibration,
    save_rule_calibration,
    load_telemetry_calibration,
    save_telemetry_calibration,
    calibrate_telemetry_model,
    StressML,
    classify_disease_group,
    disease_codes_for_profile,
    ensure_dirs,
    profile_from_group_choice,
    normalize_disease_group,
    calibrate_rule_model,
    heart_stress_score,
    hr_summary,
    now_iso,
    pbkdf2_hash,
    pbkdf2_verify,
    percent_diff,
    role_label,
    safe_json_dumps,
    safe_json_loads,
    stress_class,
    telemetry_stress_score,
    validate_password,
    validate_username,
    DISEASE_GROUP_CHOICES,
)

ensure_dirs()
DB_PATH = DATA_DIR / DB_NAME
JWT_SECRET = "stress-is-dev-secret-please-change-this-key"
JWT_ALG = "HS256"
TOKEN_EXPIRE_HOURS = 24 * 14

app = FastAPI(title=APP_NAME, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")
app.mount("/cache", StaticFiles(directory=str(CACHE_DIR)), name="cache")

_db_lock = threading.RLock()


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _execute(self, sql: str, params: tuple = ()):
        with _db_lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def _init_schema(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS profiles(
            user_id INTEGER PRIMARY KEY,
            full_name TEXT DEFAULT '',
            age INTEGER DEFAULT 0,
            sex TEXT DEFAULT '',
            healthy INTEGER DEFAULT 0,
            asthma INTEGER DEFAULT 0,
            lung INTEGER DEFAULT 0,
            heart INTEGER DEFAULT 0,
            cardio INTEGER DEFAULT 0,
            other INTEGER DEFAULT 0,
            notes TEXT DEFAULT '',
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS sessions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            researcher_id INTEGER,
            game_title TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            notes TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'started',
            age INTEGER DEFAULT 0,
            sex TEXT DEFAULT '',
            disease_group TEXT DEFAULT '',
            disease_codes TEXT DEFAULT '',
            video_path TEXT DEFAULT '',
            audio_path TEXT DEFAULT '',
            timeline_json TEXT DEFAULT '',
            summary_json TEXT DEFAULT '',
            heart_score REAL DEFAULT 0,
            telemetry_score REAL DEFAULT 0,
            ml_score REAL DEFAULT 0,
            overall_score REAL DEFAULT 0,
            stress_class TEXT DEFAULT '',
            has_rr INTEGER DEFAULT 0,
            FOREIGN KEY(player_id) REFERENCES users(id)
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS games(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS evaluations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            ts TEXT NOT NULL,
            heart_score REAL NOT NULL,
            telemetry_score REAL NOT NULL,
            ml_score REAL NOT NULL,
            overall_score REAL NOT NULL,
            stress_class TEXT NOT NULL,
            details_json TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS login_tokens(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """)
        self.conn.commit()

    def row(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        with _db_lock:
            cur = self.conn.execute(sql, params)
            return cur.fetchone()

    def rows(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        with _db_lock:
            cur = self.conn.execute(sql, params)
            return cur.fetchall()

    def execute(self, sql: str, params: tuple = ()) -> int:
        with _db_lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid

    def upsert_profile(self, user_id: int, profile: Dict[str, Any]) -> None:
        group = normalize_disease_group(profile.get("disease_group") or classify_disease_group(profile))
        values = (
            user_id,
            (profile.get("full_name") or "").strip(),
            int(profile.get("age") or 0),
            (profile.get("sex") or "").strip(),
            int(group == "healthy"),
            int(group == "asthma"),
            int(group == "asthma"),
            int(group == "heart"),
            int(group == "heart"),
            0,
            (profile.get("notes") or "").strip(),
        )
        self.execute("""
        INSERT INTO profiles(user_id, full_name, age, sex, healthy, asthma, lung, heart, cardio, other, notes)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            full_name=excluded.full_name,
            age=excluded.age,
            sex=excluded.sex,
            healthy=excluded.healthy,
            asthma=excluded.asthma,
            lung=excluded.lung,
            heart=excluded.heart,
            cardio=excluded.cardio,
            other=excluded.other,
            notes=excluded.notes
        """, values)

    def get_profile(self, user_id: int) -> Dict[str, Any]:
        row = self.row("SELECT * FROM profiles WHERE user_id=?", (user_id,))
        return dict(row) if row else {}

    def create_user(self, username: str, password: str, role: str) -> Dict[str, Any]:
        username = validate_username(username)
        password = validate_password(password)
        if role not in ROLES:
            raise ValueError("Неверная роль")
        if self.row("SELECT id FROM users WHERE username=?", (username,)):
            raise ValueError("Пользователь уже существует")
        user_id = self.execute(
            "INSERT INTO users(username, password_hash, role, created_at) VALUES(?,?,?,?)",
            (username, pbkdf2_hash(password), role, now_iso()),
        )
        self.execute(
            "INSERT OR IGNORE INTO profiles(user_id) VALUES(?)",
            (user_id,),
        )
        return self.get_user_by_id(user_id)

    def get_user_by_id(self, user_id: int) -> Dict[str, Any]:
        row = self.row("SELECT * FROM users WHERE id=?", (user_id,))
        if not row:
            return {}
        data = dict(row)
        data["profile"] = self.get_profile(user_id)
        return data

    def get_user_by_username(self, username: str) -> Dict[str, Any]:
        row = self.row("SELECT * FROM users WHERE username=?", (username,))
        if not row:
            return {}
        return self.get_user_by_id(int(row["id"]))

    def verify_login(self, username: str, password: str) -> Dict[str, Any]:
        user = self.get_user_by_username(username)
        if not user:
            raise ValueError("Неверный логин или пароль")
        if not pbkdf2_verify(password, user["password_hash"]):
            raise ValueError("Неверный логин или пароль")
        return user

    def create_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        game_title = (payload.get("game_title") or "").strip()
        if not game_title:
            raise ValueError("Не указано название игры")
        player_id = int(payload["player_id"])
        profile = profile_from_group_choice(
            payload.get("full_name") or payload.get("username") or "",
            payload.get("age") or 0,
            payload.get("sex") or "",
            payload.get("disease_group") or "healthy",
            payload.get("notes") or "",
        )
        self.upsert_profile(player_id, profile)
        profile = self.get_profile(player_id)
        disease_group = normalize_disease_group(payload.get("disease_group") or profile.get("disease_group") or classify_disease_group(profile))
        codes = payload.get("disease_codes") or disease_codes_for_profile(profile, 1)
        session_id = self.execute("""
            INSERT INTO sessions(
                player_id, researcher_id, game_title, started_at, notes, status,
                age, sex, disease_group, disease_codes
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (
            player_id,
            payload.get("researcher_id"),
            game_title,
            now_iso(),
            (payload.get("notes") or "").strip(),
            "started",
            int(payload.get("age") or profile.get("age") or 0),
            (payload.get("sex") or profile.get("sex") or "").strip(),
            disease_group,
            codes,
        ))
        self.execute("INSERT OR IGNORE INTO games(title, created_at) VALUES(?,?)", (game_title, now_iso()))
        return self.get_session(session_id)

    def update_session(self, session_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        fields = []
        values = []
        for key in ["ended_at", "notes", "status", "video_path", "audio_path", "timeline_json",
                    "summary_json", "heart_score", "telemetry_score", "ml_score", "overall_score",
                    "stress_class", "has_rr"]:
            if key in payload:
                fields.append(f"{key}=?")
                values.append(payload[key])
        if not fields:
            return self.get_session(session_id)
        values.append(session_id)
        self.execute(f"UPDATE sessions SET {', '.join(fields)} WHERE id=?", tuple(values))
        return self.get_session(session_id)

    def save_evaluation(self, session_id: int, details: Dict[str, Any]) -> None:
        self.execute("""
            INSERT INTO evaluations(session_id, ts, heart_score, telemetry_score, ml_score, overall_score, stress_class, details_json)
            VALUES(?,?,?,?,?,?,?,?)
        """, (
            session_id, now_iso(), details["heart_score"], details["telemetry_score"], details["ml_score"],
            details["overall_score"], details["stress_class"], safe_json_dumps(details),
        ))

    def get_session(self, session_id: int) -> Dict[str, Any]:
        row = self.row("""
            SELECT s.*, u.username AS player_username, u.role AS player_role
            FROM sessions s
            JOIN users u ON u.id=s.player_id
            WHERE s.id=?
        """, (session_id,))
        return dict(row) if row else {}


    def latest_session_for_player(self, player_id: int) -> Dict[str, Any]:
        row = self.row("""
            SELECT * FROM sessions
            WHERE player_id=?
            ORDER BY id DESC
            LIMIT 1
        """, (player_id,))
        return dict(row) if row else {}

    def list_sessions(self, player_id: Optional[int] = None, game_title: Optional[str] = None) -> List[Dict[str, Any]]:
        where = []
        params: List[Any] = []
        if player_id:
            where.append("s.player_id=?")
            params.append(player_id)
        if game_title:
            where.append("s.game_title=?")
            params.append(game_title)
        sql = """
            SELECT s.*, u.username AS player_username
            FROM sessions s
            JOIN users u ON u.id=s.player_id
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY s.id DESC"
        return [dict(r) for r in self.rows(sql, tuple(params))]

    def list_players(self) -> List[Dict[str, Any]]:
        rows = self.rows("""
            SELECT u.id, u.username, u.role, u.created_at,
                   COALESCE(p.full_name, '') AS full_name,
                   COALESCE(p.age, 0) AS age,
                   COALESCE(p.sex, '') AS sex,
                   COALESCE(p.healthy, 0) AS healthy,
                   COALESCE(p.asthma, 0) AS asthma,
                   COALESCE(p.lung, 0) AS lung,
                   COALESCE(p.heart, 0) AS heart,
                   COALESCE(p.cardio, 0) AS cardio,
                   COALESCE(p.other, 0) AS other,
                   COALESCE(p.notes, '') AS notes
            FROM users u
            LEFT JOIN profiles p ON p.user_id=u.id
            WHERE u.role=? AND (SELECT COUNT(*) FROM sessions WHERE player_id=u.id) > 0
            ORDER BY u.id DESC
        """, (ROLE_PLAYER,))
        out = []
        counters = {"healthy": 0, "asthma": 0, "heart": 0}
        for r in rows:
            data = dict(r)
            latest = self.latest_session_for_player(int(data["id"]))
            # Если профиль ещё не заполнен, берём параметры из последней сессии.
            if not int(data.get("age") or 0) and latest:
                data["age"] = int(latest.get("age") or 0)
            if not str(data.get("sex") or "").strip() and latest:
                data["sex"] = latest.get("sex") or ""
            profile_like = {
                "disease_group": data.get("disease_group", ""),
                "healthy": data.get("healthy", 0),
                "asthma": data.get("asthma", 0),
                "lung": data.get("lung", 0),
                "heart": data.get("heart", 0),
                "cardio": data.get("cardio", 0),
                "other": data.get("other", 0),
            }
            if latest:
                latest_group = normalize_disease_group(latest.get("disease_group") or "")
                if latest_group in DISEASE_GROUP_CHOICES:
                    profile_like["disease_group"] = latest_group
                    profile_like["healthy"] = int(latest_group == "healthy")
                    profile_like["asthma"] = int(latest_group == "asthma")
                    profile_like["lung"] = int(latest_group == "asthma")
                    profile_like["heart"] = int(latest_group == "heart")
                    profile_like["cardio"] = int(latest_group == "heart")
            try:
                group = classify_disease_group(profile_like)
            except Exception:
                group = "healthy"
            counters[group] = counters.get(group, 0) + 1
            data["disease_group"] = group
            data["disease_code"] = disease_codes_for_profile(profile_like, counters[group])
            data["group_index"] = counters[group]
            out.append(data)
        return out

    def list_games(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.rows("SELECT * FROM games ORDER BY title")]

    def comparison_by_game(self, game_title: str) -> Dict[str, Any]:
        sessions = self.rows("""
            SELECT s.*, u.username, COALESCE(p.full_name, '') AS full_name,
                   COALESCE(p.age, 0) AS age, COALESCE(p.sex, '') AS sex,
                   COALESCE(p.healthy, 0) AS healthy, COALESCE(p.asthma, 0) AS asthma,
                   COALESCE(p.lung, 0) AS lung, COALESCE(p.heart, 0) AS heart,
                   COALESCE(p.cardio, 0) AS cardio, COALESCE(p.other, 0) AS other
            FROM sessions s
            JOIN users u ON u.id=s.player_id
            LEFT JOIN profiles p ON p.user_id=u.id
            WHERE s.game_title=?
        """, (game_title,))
        groups: Dict[str, List[sqlite3.Row]] = {"healthy": [], "asthma": [], "heart": []}
        for s in sessions:
            grp = classify_disease_group(dict(s))
            groups.setdefault(grp, []).append(s)

        def avg(field: str, rows: List[sqlite3.Row]) -> float:
            vals = [float(r[field]) for r in rows if r[field] is not None]
            return sum(vals)/len(vals) if vals else 0.0

        summary = {}
        for grp in ("healthy", "asthma", "heart"):
            rows = groups.get(grp, [])
            summary[grp] = {
                "count": len(rows),
                "overall_avg": avg("overall_score", rows),
                "heart_avg": avg("heart_score", rows),
                "telemetry_avg": avg("telemetry_score", rows),
                "ml_avg": avg("ml_score", rows),
            }
        healthy = summary["healthy"]["overall_avg"] or 1.0
        for grp, info in summary.items():
            info["vs_healthy_percent"] = percent_diff(info["overall_avg"], healthy)
        return {"game_title": game_title, "groups": summary, "sessions": [dict(s) for s in sessions]}

db = Database(DB_PATH)
ml = StressML()

def make_token(user: Dict[str, Any]) -> str:
    expires = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": str(user["id"]),
        "username": user["username"],
        "role": user["role"],
        "exp": expires,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    db.execute("INSERT INTO login_tokens(user_id, token, expires_at) VALUES(?,?,?)", (user["id"], token, expires.isoformat()))
    return token

def auth_user(authorization: str = Header(default="")) -> Dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        user_id = int(payload["sub"])
        user = db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="Пользователь не найден")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Срок токена истёк")
    except Exception:
        raise HTTPException(status_code=401, detail="Неверный токен")

class RegisterIn(BaseModel):
    username: str
    password: str
    role: str = Field(pattern="^(player|researcher)$")

class LoginIn(BaseModel):
    username: str
    password: str

class ProfileIn(BaseModel):
    full_name: str = ""
    age: int = 0
    sex: str = ""
    disease_group: str = "healthy"
    healthy: bool = False
    asthma: bool = False
    lung: bool = False
    heart: bool = False
    cardio: bool = False
    other: bool = False
    notes: str = ""

class SessionStartIn(BaseModel):
    game_title: str
    notes: str = ""
    full_name: str = ""
    age: int = 0
    sex: str = ""
    disease_group: str = "healthy"
    disease_codes: str = ""

class SessionFinalizeIn(BaseModel):
    timeline: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    heart_score: float = 0.0
    telemetry_score: float = 0.0
    ml_score: float = 0.0
    overall_score: float = 0.0
    stress_class: str = ""
    has_rr: bool = False
    notes: str = ""
    ended_at: str = ""
    video_path: str = ""
    audio_path: str = ""

@app.get("/health")
def health():
    return {"status": "ok", "name": APP_NAME}

@app.post("/auth/register")
def register(payload: RegisterIn):
    try:
        user = db.create_user(payload.username, payload.password, payload.role)
        token = make_token(user)
        return {"token": token, "user": _user_payload(user)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
def login(payload: LoginIn):
    try:
        user = db.verify_login(payload.username, payload.password)
        token = make_token(user)
        return {"token": token, "user": _user_payload(user)}
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.get("/me")
def me(authorization: str = Header(default="")):
    user = auth_user(authorization)
    return _user_payload(user)

@app.post("/profile")
def update_profile(payload: ProfileIn, authorization: str = Header(default="")):
    user = auth_user(authorization)
    db.upsert_profile(user["id"], payload.model_dump())
    return {"ok": True, "profile": db.get_profile(user["id"])}

@app.get("/players")
def players(authorization: str = Header(default="")):
    user = auth_user(authorization)
    if user["role"] != ROLE_RESEARCHER:
        return [_player_payload(db.get_user_by_id(user["id"]))]
    # Для исследователя возвращаем уже агрегированные записи без повторной
    # нормализации, чтобы не терять возраст, пол, группу и код.
    return db.list_players()

@app.get("/players/{player_id}")
def player_detail(player_id: int, authorization: str = Header(default="")):
    user = auth_user(authorization)
    if user["role"] != ROLE_RESEARCHER and user["id"] != player_id:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    pdata = db.get_user_by_id(player_id)
    if not pdata:
        raise HTTPException(status_code=404, detail="Игрок не найден")
    pdata["sessions"] = db.list_sessions(player_id=player_id)
    return _player_payload(pdata, include_sessions=True)

@app.get("/games")
def games(authorization: str = Header(default="")):
    user = auth_user(authorization)
    if user["role"] == ROLE_PLAYER:
        sessions = db.list_sessions(player_id=user["id"])
        titles = []
        seen = set()
        for s in sessions:
            title = s["game_title"]
            if title not in seen:
                seen.add(title)
                titles.append({"title": title})
        return titles
    return db.list_games()

@app.get("/games/{game_title}/sessions")
def game_sessions(game_title: str, authorization: str = Header(default="")):
    user = auth_user(authorization)
    sessions = db.list_sessions(game_title=game_title)
    if user["role"] != ROLE_RESEARCHER:
        sessions = [s for s in sessions if s["player_id"] == user["id"]]
    return sessions

@app.get("/sessions")
def sessions(authorization: str = Header(default=""), player_id: Optional[int] = None, game_title: Optional[str] = None):
    user = auth_user(authorization)
    if user["role"] != ROLE_RESEARCHER:
        player_id = user["id"]
    return db.list_sessions(player_id=player_id, game_title=game_title)

@app.get("/sessions/{session_id}")
def session_detail(session_id: int, authorization: str = Header(default="")):
    user = auth_user(authorization)
    s = db.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    if user["role"] != ROLE_RESEARCHER and s["player_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    s["timeline"] = safe_json_loads(s.get("timeline_json"), [])
    s["summary"] = safe_json_loads(s.get("summary_json"), {})
    s["player"] = _player_payload(db.get_user_by_id(s["player_id"]))
    return s

@app.get("/sessions/{session_id}/timeline")
def session_timeline(session_id: int, authorization: str = Header(default="")):
    user = auth_user(authorization)
    s = db.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    if user["role"] != ROLE_RESEARCHER and s["player_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    return {"timeline": safe_json_loads(s.get("timeline_json"), []), "summary": safe_json_loads(s.get("summary_json"), {})}

@app.get("/sessions/{session_id}/video")
def session_video(session_id: int, authorization: str = Header(default="")):
    user = auth_user(authorization)
    s = db.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    if user["role"] != ROLE_RESEARCHER and s["player_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    if not s.get("video_path"):
        raise HTTPException(status_code=404, detail="Видео не загружено")
    return FileResponse(s["video_path"])

@app.post("/sessions")
def create_session(payload: SessionStartIn, authorization: str = Header(default="")):
    user = auth_user(authorization)
    if user["role"] != ROLE_PLAYER:
        raise HTTPException(status_code=403, detail="Создавать сессии может только игрок")
    session = db.create_session({"player_id": user["id"], **payload.model_dump()})
    return session

@app.put("/sessions/{session_id}")
def finalize_session(session_id: int, payload: SessionFinalizeIn, authorization: str = Header(default="")):
    user = auth_user(authorization)
    s = db.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    if user["role"] != ROLE_RESEARCHER and s["player_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    timeline_json = safe_json_dumps(payload.timeline)
    summary_json = safe_json_dumps(payload.summary)
    updated = db.update_session(session_id, {
        "ended_at": payload.ended_at or now_iso(),
        "status": "finished",
        "notes": payload.notes or s.get("notes", ""),
        "timeline_json": timeline_json,
        "summary_json": summary_json,
        "heart_score": payload.heart_score,
        "telemetry_score": payload.telemetry_score,
        "ml_score": payload.ml_score,
        "overall_score": payload.overall_score,
        "stress_class": payload.stress_class,
        "has_rr": int(bool(payload.has_rr)),
        "video_path": payload.video_path or s.get("video_path", ""),
        "audio_path": payload.audio_path or s.get("audio_path", ""),
    })
    db.save_evaluation(session_id, {
        "heart_score": payload.heart_score,
        "telemetry_score": payload.telemetry_score,
        "ml_score": payload.ml_score,
        "overall_score": payload.overall_score,
        "stress_class": payload.stress_class,
        "timeline_len": len(payload.timeline),
        "summary": payload.summary,
    })
    return updated

@app.post("/sessions/{session_id}/video")
async def upload_video(session_id: int, file: UploadFile = File(...), authorization: str = Header(default="")):
    user = auth_user(authorization)
    s = db.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    if user["role"] != ROLE_RESEARCHER and s["player_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    suffix = Path(file.filename or "session.mp4").suffix.lower() or ".mp4"
    out = MEDIA_DIR / f"session_{session_id}{suffix}"
    data = await file.read()
    out.write_bytes(data)
    db.update_session(session_id, {"video_path": str(out)})
    return {"ok": True, "path": str(out), "url": f"/media/{out.name}"}

@app.post("/sessions/{session_id}/audio")
async def upload_audio(session_id: int, file: UploadFile = File(...), authorization: str = Header(default="")):
    user = auth_user(authorization)
    s = db.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    if user["role"] != ROLE_RESEARCHER and s["player_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    suffix = Path(file.filename or "session.wav").suffix.lower() or ".wav"
    out = MEDIA_DIR / f"session_{session_id}{suffix}"
    data = await file.read()
    out.write_bytes(data)
    db.update_session(session_id, {"audio_path": str(out)})
    return {"ok": True, "path": str(out), "url": f"/media/{out.name}"}

@app.get("/research/comparison/{game_title}")
def research_comparison(game_title: str, authorization: str = Header(default="")):
    user = auth_user(authorization)
    if user["role"] != ROLE_RESEARCHER:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    return db.comparison_by_game(game_title)

@app.get("/research/players")
def research_players(authorization: str = Header(default="")):
    user = auth_user(authorization)
    if user["role"] != ROLE_RESEARCHER:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    return db.list_players()

@app.get("/research/players/{player_id}/sessions")
def research_player_sessions(player_id: int, authorization: str = Header(default="")):
    user = auth_user(authorization)
    if user["role"] != ROLE_RESEARCHER:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    return db.list_sessions(player_id=player_id)

@app.get("/research/games")
def research_games(authorization: str = Header(default="")):
    user = auth_user(authorization)
    if user["role"] != ROLE_RESEARCHER:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    return db.list_games()

@app.get("/research/train-model")
def research_train_model(authorization: str = Header(default="")):
    user = auth_user(authorization)
    if user["role"] not in {ROLE_PLAYER, ROLE_RESEARCHER}:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    rows = db.rows("SELECT * FROM evaluations")
    if len(rows) < 2:
        return {"ok": True, "trained": False, "message": "Недостаточно данных, используется текущая модель", "model": str(ml.model_path)}
    X = []
    y = []
    for r in rows:
        details = safe_json_loads(r["details_json"], {})
        X.append(details)
        y.append(1 if r["stress_class"] == "высокий" else 0)
    try:
        ml.fit(X, y)
        return {"ok": True, "trained": True, "model": str(ml.model_path)}
    except Exception as e:
        return {"ok": False, "trained": False, "message": f"Не удалось обучить модель: {e}", "model": str(ml.model_path)}


@app.post("/ml/predict")
def ml_predict(payload: Dict[str, Any], authorization: str = Header(default="")):
    user = auth_user(authorization)
    if user["role"] not in {ROLE_PLAYER, ROLE_RESEARCHER}:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    return {"ml_prob": ml.predict_prob(payload)}


@app.get("/calibration/rule")
def rule_calibration_status(authorization: str = Header(default="")):
    user = auth_user(authorization)
    if user["role"] not in {ROLE_PLAYER, ROLE_RESEARCHER}:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    return load_rule_calibration()


@app.post("/calibration/rule")
def rule_calibration_train(authorization: str = Header(default="")):
    user = auth_user(authorization)
    if user["role"] not in {ROLE_PLAYER, ROLE_RESEARCHER}:
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    rows = db.rows("SELECT * FROM evaluations")
    if len(rows) < 2:
        calib = load_rule_calibration()
        calib.update({"calibrated": False, "source_rows": len(rows), "message": "Недостаточно данных, оставлены текущие веса"})
        return calib
    try:
        feats = []
        labels = []
        for r in rows:
            details = safe_json_loads(r.get("details_json"), {})
            ctx = details if isinstance(details, dict) else {}
            feats.append({
                "heart_score": float(r["heart_score"]),
                "telemetry_score": float(r["telemetry_score"]),
                "ml_score": float(r["ml_score"]),
                **ctx,
            })
            labels.append(1 if r["stress_class"] == "высокий" else 0)
        calib = calibrate_rule_model(feats, labels)
        save_rule_calibration(calib)
        telemetry_calib = calibrate_telemetry_model(feats, labels)
        save_telemetry_calibration(telemetry_calib)
        return {**calib, "telemetry_calibration": telemetry_calib}
    except Exception as e:
        calib = load_rule_calibration()
        calib.update({"calibrated": False, "source_rows": len(rows), "message": f"Калибровка не выполнена: {e}"})
        return calib



def _user_payload(user: Dict[str, Any]) -> Dict[str, Any]:
    profile = user.get("profile") or {}
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "role_label": role_label(user["role"]),
        "profile": profile,
    }

def _player_payload(user: Dict[str, Any], include_sessions: bool = False) -> Dict[str, Any]:
    if not user:
        return {}
    profile = dict(user.get("profile") or {})
    latest = user.get("latest_session") or {}
    # Если пришла уже агрегированная запись (из /players для исследователя),
    # не теряем готовые поля возраста/пола/группы/кода.
    age_raw = user.get("age", profile.get("age", 0))
    sex_raw = user.get("sex", profile.get("sex", ""))
    group_raw = user.get("disease_group", profile.get("disease_group", ""))
    code_raw = user.get("disease_code", "")
    if not int(age_raw or 0) and latest:
        age_raw = latest.get("age") or 0
    if not str(sex_raw or "").strip() and latest:
        sex_raw = latest.get("sex") or ""
    if not str(group_raw or "").strip() and latest:
        group_raw = latest.get("disease_group") or ""
    if not code_raw:
        code_raw = user.get("disease_code") or (disease_codes_for_profile(profile, 1) if profile else "")
    if not str(group_raw or "").strip():
        group_raw = classify_disease_group(profile or latest)
    data = {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "role_label": role_label(user["role"]),
        "profile": profile,
        "age": int(age_raw or 0),
        "sex": sex_raw or "",
        "disease_group": group_raw,
        "disease_code": code_raw or disease_codes_for_profile(profile or {"disease_group": group_raw}, 1),
    }
    if include_sessions:
        data["sessions"] = user.get("sessions", [])
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_app:app", host="0.0.0.0", port=8000, reload=False)
