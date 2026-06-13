import pytest

def test_register_and_login(client):
    resp = client.post("/auth/register", json={
        "username": "func_user",
        "password": "func_pass",
        "role": "player"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "token" in data
    assert data["user"]["username"] == "func_user"

    resp = client.post("/auth/login", json={
        "username": "func_user",
        "password": "func_pass"
    })
    assert resp.status_code == 200
    assert "token" in resp.json()

def test_profile_update(client, auth_headers):
    resp = client.post("/profile", headers=auth_headers, json={
        "full_name": "Test User",
        "age": 30,
        "sex": "male",
        "disease_group": "asthma"
    })
    assert resp.status_code == 200
    me = client.get("/me", headers=auth_headers)
    assert me.json()["profile"]["full_name"] == "Test User"

def test_create_and_finalize_session(client, auth_headers):
    start_payload = {
        "game_title": "Test Game",
        "notes": "functional test",
        "full_name": "Tester",
        "age": 25,
        "sex": "female",
        "disease_group": "healthy"
    }
    resp = client.post("/sessions", headers=auth_headers, json=start_payload)
    assert resp.status_code == 200
    session_id = resp.json()["id"]

    finalize_payload = {
        "timeline": [{"t": 1.0, "overall_score": 0.6}],
        "summary": {"overall": 0.6},
        "heart_score": 0.5,
        "telemetry_score": 0.4,
        "ml_score": 0.55,
        "overall_score": 0.6,
        "stress_class": "средний",
        "has_rr": False,
        "notes": "",
        "ended_at": "2026-06-13T12:00:00Z",
        "video_path": "",
        "audio_path": ""
    }
    resp = client.put(f"/sessions/{session_id}", headers=auth_headers, json=finalize_payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "finished"

def test_ml_predict(client, auth_headers):
    payload = {
        "hr_mean": 85.0,
        "mouse_speed_mean": 400.0,
        "error_rate": 0.05
    }
    resp = client.post("/ml/predict", headers=auth_headers, json=payload)
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["ml_prob"] <= 1.0

def test_rule_calibration(client, auth_headers):
    resp = client.post("/calibration/rule", headers=auth_headers)
    assert resp.status_code == 200
    assert "weights" in resp.json()

def test_train_model(client, auth_headers):
    resp = client.get("/research/train-model", headers=auth_headers)
    assert resp.status_code == 200
    assert "model" in resp.json()