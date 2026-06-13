import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from server_app import app, db
import tempfile
import shutil
import sqlite3


@pytest.fixture(scope="function")
def test_db():
    """Временная БД для каждого теста, с корректным закрытием соединений."""
    tmp_dir = tempfile.mkdtemp()
    db_path = Path(tmp_dir) / "test_stress_is.sqlite3"

    # Создаём новый экземпляр Database
    from server_app import Database
    test_db_instance = Database(db_path)

    # Сохраняем оригинальную БД и подменяем
    original_db = db
    import server_app
    server_app.db = test_db_instance

    yield test_db_instance

    # Закрываем соединение перед удалением
    test_db_instance.conn.close()
    server_app.db = original_db

    # Удаляем временную папку
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def client(test_db):
    """HTTP клиент для тестирования API."""
    return TestClient(app)


@pytest.fixture
def auth_headers(client):
    """Регистрирует и возвращает заголовки авторизации для игрока."""
    r = client.post("/auth/register", json={
        "username": "testplayer",
        "password": "test123",
        "role": "player"
    })
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}"}