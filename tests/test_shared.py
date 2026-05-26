from fastapi.testclient import TestClient

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from shared import (
    baevsky_stress_index,
    combine_stress_scores,
    classify_disease_group,
    profile_from_group_choice,
    disease_codes_for_profile,
    pbkdf2_hash,
    pbkdf2_verify,
    rmssd,
    sdnn,
    telemetry_summary,
    telemetry_stress_score,
    validate_password,
    validate_username,
)
from server_app import app


def test_username_validation():
    try:
        validate_username("")
        assert False, "должна быть ошибка"
    except ValueError:
        pass
    assert validate_username("user01") == "user01"


def test_password_validation():
    try:
        validate_password("")
        assert False, "должна быть ошибка"
    except ValueError:
        pass
    assert validate_password("pass123") == "pass123"


def test_password_hash_roundtrip():
    h = pbkdf2_hash("secret")
    assert pbkdf2_verify("secret", h)
    assert not pbkdf2_verify("wrong", h)


def test_hrv_formulas():
    rr = [800, 810, 790, 805, 820]
    assert sdnn(rr) > 0
    assert rmssd(rr) > 0
    assert baevsky_stress_index(rr) >= 0


def test_disease_codes():
    assert disease_codes_for_profile({"healthy": True}, 2) == "З2"
    assert disease_codes_for_profile({"asthma": True}, 1) == "Л1"
    assert disease_codes_for_profile({"heart": True}, 3) == "С3"


def test_api_register_login_and_profile():
    client = TestClient(app)
    uname = "tester_" + __import__("uuid").uuid4().hex[:10]
    r = client.post("/auth/register", json={"username": uname, "password": "pass1234", "role": "player"})
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    h = {"Authorization": f"Bearer {token}"}
    r2 = client.post("/profile", headers=h, json={
        "full_name": "Тестовый Игрок",
        "age": 21,
        "sex": "мужской",
        "healthy": True
    })
    assert r2.status_code == 200, r2.text
    me = client.get("/me", headers=h)
    assert me.status_code == 200

def test_player_group_and_profile_choice():
    prof = profile_from_group_choice("Иван Иванов", 24, "женский", "heart", "")
    assert classify_disease_group(prof) == "heart"
    assert prof["age"] == 24
    assert prof["sex"] == "женский"


def test_research_players_preserve_profile_values():
    client = TestClient(app)
    uname = "prof_" + __import__("uuid").uuid4().hex[:10]
    rname = "research_" + __import__("uuid").uuid4().hex[:10]
    p = client.post("/auth/register", json={"username": uname, "password": "pass1234", "role": "player"})
    r = client.post("/auth/register", json={"username": rname, "password": "pass1234", "role": "researcher"})
    ph = {"Authorization": f"Bearer {p.json()['token']}"}
    rh = {"Authorization": f"Bearer {r.json()['token']}"}
    prof = {"full_name": "Алина Петрова", "age": 27, "sex": "женский", "disease_group": "asthma", "notes": ""}
    assert client.post("/profile", headers=ph, json=prof).status_code == 200
    players = client.get("/players", headers=rh)
    assert players.status_code == 200, players.text
    rows = players.json()
    row = next(x for x in rows if x["username"] == uname)
    assert row["age"] == 27
    assert row["sex"] == "женский"
    assert row["disease_group"] == "asthma"
    assert row["disease_code"].startswith("Л")



def test_player_list_and_calibration_endpoints():
    client = TestClient(app)
    pname = "p_" + __import__("uuid").uuid4().hex[:8]
    rname = "r_" + __import__("uuid").uuid4().hex[:8]
    p = client.post("/auth/register", json={"username": pname, "password": "pass1234", "role": "player"})
    r = client.post("/auth/register", json={"username": rname, "password": "pass1234", "role": "researcher"})
    ph = {"Authorization": f"Bearer {p.json()['token']}"}
    rh = {"Authorization": f"Bearer {r.json()['token']}"}

    players = client.get("/players", headers=rh)
    assert players.status_code == 200, players.text
    assert isinstance(players.json(), list)

    calib = client.post("/calibration/rule", headers=ph)
    assert calib.status_code == 200, calib.text
    assert "weights" in calib.json()

    train = client.get("/research/train-model", headers=ph)
    assert train.status_code == 200, train.text
    assert "model" in train.json()


def test_telemetry_features_and_combination():
    events = [
        {"t": 0.0, "type": "mouse_move", "x": 10, "y": 10, "dist": 4.0, "speed": 400.0, "accel": 1500.0, "jerk": 2400.0},
        {"t": 0.1, "type": "mouse_move", "x": 20, "y": 5, "dist": 12.0, "speed": 1200.0, "accel": 2800.0, "jerk": 4200.0},
        {"t": 0.2, "type": "click", "button": "left"},
        {"t": 0.25, "type": "rapid_click", "button": "left"},
        {"t": 0.4, "type": "backspace", "key": "Backspace"},
        {"t": 0.5, "type": "escape", "key": "Esc"},
    ]
    feat = telemetry_summary(events, window_seconds=10.0)
    assert feat["error_rate"] > 0
    assert feat["mouse_spike_rate"] > 0 or feat["mouse_reversal_rate"] >= 0
    score = telemetry_stress_score(feat)
    assert 0.0 <= score <= 1.0
    assert combine_stress_scores(None, score, None) == score



def test_login_rejects_wrong_password():
    client = TestClient(app)
    uname = "bad_" + __import__("uuid").uuid4().hex[:10]
    r = client.post("/auth/register", json={"username": uname, "password": "pass1234", "role": "player"})
    assert r.status_code == 200, r.text
    bad = client.post("/auth/login", json={"username": uname, "password": "wrongpass"})
    assert bad.status_code == 401, bad.text
    assert "Неверный логин или пароль" in bad.text
