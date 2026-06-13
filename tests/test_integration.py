import pytest
from unittest.mock import patch, MagicMock
import time

def test_full_session_flow(client, auth_headers):
    """Сквозной сценарий через реальный API."""
    # Создание сессии
    start = {
        "game_title": "Integration Test Game",
        "notes": "testing",
        "full_name": "Integ Tester",
        "age": 28,
        "sex": "male",
        "disease_group": "healthy"
    }
    resp = client.post("/sessions", headers=auth_headers, json=start)
    assert resp.status_code == 200
    session_id = resp.json()["id"]

    # Вызов ML предсказания с тестовыми данными
    telemetry = {
        "mouse_speed_mean": 300.0,
        "click_rate": 1.0,
        "error_rate": 0.0
    }
    ml_resp = client.post("/ml/predict", headers=auth_headers, json=telemetry)
    assert ml_resp.status_code == 200
    ml_prob = ml_resp.json()["ml_prob"]

    # Завершение сессии
    finalize = {
        "timeline": [{"t": 1.0, "overall_score": 0.45}],
        "summary": {"overall": 0.45},
        "heart_score": 0.4,
        "telemetry_score": 0.5,
        "ml_score": ml_prob,
        "overall_score": 0.45,
        "stress_class": "средний",
        "has_rr": False,
        "notes": "",
        "ended_at": "",
        "video_path": "",
        "audio_path": ""
    }
    resp = client.put(f"/sessions/{session_id}", headers=auth_headers, json=finalize)
    assert resp.status_code == 200
    assert resp.json()["status"] == "finished"

    # Проверка, что сессия появилась в списке
    sessions = client.get("/sessions", headers=auth_headers).json()
    assert any(s["id"] == session_id for s in sessions)

def test_integration_with_heart_sensor(client, auth_headers):
    """Проверка, что данные с датчика корректно парсятся и могут быть отправлены."""
    from desktop_client import HeartSensorReader
    import threading
    import time

    # Мокаем serial порт
    with patch("serial.Serial") as mock_serial:
        mock_serial_instance = MagicMock()
        # Симулируем чтение строки с датчика
        mock_serial_instance.readline.return_value = b"IBI: 842 BPM: 71\r\n"
        mock_serial.return_value = mock_serial_instance

        sensor = HeartSensorReader()
        sensor.connect("COM3")
        # Ждём, пока поток обработает
        time.sleep(0.3)
        packet = sensor.sample_packet()
        assert packet["ibi_ms"] == 842.0 or packet["bpm"] == 71.0
        sensor.disconnect()