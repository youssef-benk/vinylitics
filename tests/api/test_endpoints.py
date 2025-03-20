import pytest
from httpx import AsyncClient
import os

test_params = {
    'track_name': 'Shape of you',
    'artist': 'Ed Sheeran'
}

@pytest.mark.asyncio
async def test_root_is_up():
    from vinylitics.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_root_returns_greeting():
    from vinylitics.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.json() == {"greeting": "Welcome to Vynilitics API"}

@pytest.mark.asyncio
async def test_predict_is_up():
    from vinylitics.api.fast import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/predict_spotify", params=test_params)
    assert response.status_code == 200
