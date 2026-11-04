"""hass_get_state 交回的亮度按百分比报（HA 的 brightness 是 0～255，原样报会被念成"一百五十三"）。
HTTP 层换成假的客户端，不出网。"""
import asyncio

import pytest

from plugins_func.functions import hass_state as hass_module


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


class FakeAsyncClient:
    payload = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, headers=None):
        return FakeResponse(FakeAsyncClient.payload)


class FakeConn:
    def __init__(self):
        self.load_function_plugin = True
        self.config = {"plugins": {"home_assistant": {"base_url": "http://homeassistant.local:8123",
                                                      "api_key": "test-token"}}}


@pytest.fixture
def fake_http(monkeypatch):
    monkeypatch.setattr(hass_module.httpx, "AsyncClient", FakeAsyncClient)
    return FakeAsyncClient


def _get_state(fake_http, attributes, state="on"):
    fake_http.payload = {"state": state, "attributes": attributes}
    return asyncio.run(hass_module.handle_hass_get_state(FakeConn(), "light.study_lamp"))


@pytest.mark.parametrize("raw, shown", [(153, "60%"), (255, "100%"), (128, "50%"), (1, "1%"), (0, "0%")])
def test_brightness_reported_as_percent(fake_http, raw, shown):
    text = _get_state(fake_http, {"brightness": raw, "color_temp_kelvin": 4000})
    assert f"亮度是:{shown} " in text
    assert text.startswith("设备状态:on ") and "色温是:4000 " in text


def test_light_off_has_no_brightness_line(fake_http):
    # 灯关着时 HA 给 brightness: null，以前会报"亮度是:None"
    text = _get_state(fake_http, {"brightness": None}, state="off")
    assert text == "设备状态:off "
    assert "亮度" not in _get_state(fake_http, {"friendly_name": "书房灯"})


def test_brightness_set_is_taken_as_brightness_value(monkeypatch):
    sent = []

    class Client(FakeAsyncClient):
        async def post(self, url, headers=None, json=None):
            sent.append((url.rsplit("/services/", 1)[1], json))
            return FakeResponse({})

    monkeypatch.setattr(hass_module.httpx, "AsyncClient", Client)
    for kind in ("brightness_value", "brightness_set"):
        text = asyncio.run(hass_module.handle_hass_set_state(FakeConn(), "light.study_lamp", {"type": kind, "input": 50}))
        assert text == "亮度已调整到50"
    assert sent == [("light/turn_on", {"entity_id": "light.study_lamp", "brightness_pct": 50})] * 2
