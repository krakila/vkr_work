def test_login_wrong_password(client):
    client.post("/auth/register", json={
        "username": "neg_user", "password": "correct", "role": "player"
    })
    resp = client.post("/auth/login", json={
        "username": "neg_user", "password": "wrong"
    })
    assert resp.status_code == 401
    assert "Неверный логин или пароль" in resp.text

def test_create_session_without_auth(client):
    resp = client.post("/sessions", json={"game_title": "No Auth"})
    assert resp.status_code == 401

def test_access_other_player_session(client):
    # Игрок A
    r1 = client.post("/auth/register", json={"username": "playerA", "password": "pass", "role": "player"})
    tokenA = r1.json()["token"]
    # Игрок B
    r2 = client.post("/auth/register", json={"username": "playerB", "password": "pass", "role": "player"})
    tokenB = r2.json()["token"]

    # Сессия игрока A
    sess = client.post("/sessions", headers={"Authorization": f"Bearer {tokenA}"},
                       json={"game_title": "A's game"}).json()
    sess_id = sess["id"]

    # Попытка доступа от B
    resp = client.get(f"/sessions/{sess_id}", headers={"Authorization": f"Bearer {tokenB}"})
    assert resp.status_code == 403

def test_invalid_username_validation(client):
    resp = client.post("/auth/register", json={"username": "ab", "password": "pass123", "role": "player"})
    assert resp.status_code == 400
    assert "минимум 3 символа" in resp.text

def test_empty_password(client):
    resp = client.post("/auth/register", json={"username": "validuser", "password": "", "role": "player"})
    assert resp.status_code == 400

def test_session_creation_missing_game_title(client, auth_headers):
    resp = client.post("/sessions", headers=auth_headers, json={"notes": "no title"})
    # FastAPI возвращает 422 при ошибке валидации Pydantic
    assert resp.status_code == 422