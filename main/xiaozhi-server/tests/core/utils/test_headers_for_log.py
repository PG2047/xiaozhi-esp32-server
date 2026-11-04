"""设备身份和凭据不进日志：握手那行请求头去掉 client-id / device-id / authorization，其他字段照旧、self.headers 不动；
OTA 的 INFO / DEBUG 日志也不带 client-id。值都是假的。"""
from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

from multidict import CIMultiDict
from websockets.datastructures import Headers

from tests.conftest import CONFIG
from config.logger import setup_logging
from core.utils.cache.manager import CacheType, cache_manager

# setup_logging() 没有配置缓存时会 asyncio.run(load_config())，在事件循环里会炸；生产由 app.py 填缓存，测试自己填
cache_manager.set(CacheType.CONFIG, "main_config", CONFIG)
setup_logging()

from core.api.ota_handler import OTAHandler  # noqa: E402
from core.connection import ConnectionHandler  # noqa: E402
from core.utils.util import headers_for_log  # noqa: E402
import core.connection as connection_mod  # noqa: E402

MAC = "aa:bb:cc:dd:ee:01"
CLIENT_ID = "11111111-2222-4333-8444-555555555555"
TOKEN = "Bearer not-a-real-token"
HEADERS = {
    "host": "127.0.0.1:8000",
    "protocol-version": "1",
    "x-real-ip": "203.0.113.7",
    "device-id": MAC,
    "client-id": CLIENT_ID,
    "authorization": TOKEN,
}
HIDDEN_NAMES = ("client-id", "device-id", "authorization")
HIDDEN_VALUES = (MAC, CLIENT_ID, TOKEN)


class FakeLogger:
    def __init__(self):
        self.lines = []

    def bind(self, **_):
        return self

    def _record(self, message, *_args, **_kwargs):
        self.lines.append(str(message))

    info = debug = warning = error = _record


def assert_nothing_hidden(text: str):
    for s in HIDDEN_NAMES:
        assert s not in text.lower()
    for s in HIDDEN_VALUES:
        assert s not in text


def test_drops_identity_and_keeps_other_fields():
    assert headers_for_log(HEADERS) == {
        "host": "127.0.0.1:8000",
        "protocol-version": "1",
        "x-real-ip": "203.0.113.7",
    }


def test_header_names_match_in_any_case():
    out = headers_for_log({"Client-Id": CLIENT_ID, "DEVICE-ID": MAC, "Authorization": TOKEN, "Protocol-Version": "1"})
    assert out == {"Protocol-Version": "1"}


def test_input_is_not_changed_and_missing_names_are_fine():
    headers = dict(HEADERS)
    headers_for_log(headers)
    assert headers == HEADERS
    assert headers_for_log({"host": "x"}) == {"host": "x"}
    assert headers_for_log({}) == {}


def test_accepts_the_header_types_of_both_servers():
    # websockets 的握手请求头和 aiohttp 的 OTA 请求头
    for headers in (Headers(HEADERS), CIMultiDict(HEADERS)):
        out = headers_for_log(headers)
        assert_nothing_hidden(str(out))
        assert out["protocol-version"] == "1"


def test_handshake_log_line_has_no_identity_but_headers_stay_usable(monkeypatch):
    conn = ConnectionHandler(copy.deepcopy(CONFIG), None, None, None, None, None)
    conn.logger = FakeLogger()
    seen = {}

    def stop_after_handshake_log(c):
        seen["device_id"] = c.device_id
        raise RuntimeError("测试到这里就够了")

    async def no_close(_ws):
        pass

    monkeypatch.setattr(connection_mod, "apply_device_overrides", stop_after_handshake_log)
    monkeypatch.setattr(conn, "_save_and_close", no_close)
    ws = SimpleNamespace(request=SimpleNamespace(headers=dict(HEADERS), path="/xiaozhi/v1/"), remote_address=("127.0.0.1", 1))

    asyncio.run(conn.handle_connection(ws))

    line = next(l for l in conn.logger.lines if "conn - Headers" in l)
    assert_nothing_hidden(line)
    assert "protocol-version" in line and "x-real-ip" in line
    assert line.startswith("203.0.113.7 ")
    # 后面取设备身份还要靠 self.headers，只有日志里的副本去掉
    assert seen["device_id"] == MAC
    assert conn.headers["client-id"] == CLIENT_ID and conn.headers["authorization"] == TOKEN


def _ota_post(headers: dict) -> tuple[dict, str]:
    config = copy.deepcopy(CONFIG)
    config["server"]["auth_key"] = "not-a-real-key"    # 生产由 app.py 启动时填
    handler = OTAHandler(config)
    handler.logger = FakeLogger()

    async def text():
        return "{}"

    request = SimpleNamespace(method="POST", headers=CIMultiDict(headers), text=text)
    response = asyncio.run(handler.handle_post(request))
    return json.loads(response.text), "\n".join(handler.logger.lines)


def test_ota_log_has_no_client_id_or_header_dump():
    body, logged = _ota_post(HEADERS)
    assert "websocket" in body
    assert CLIENT_ID not in logged and TOKEN not in logged
    assert "client-id" not in logged.lower() and "ClientID" not in logged
    assert "protocol-version" in logged    # DEBUG 那行请求头还在，只是去掉了身份


def test_ota_still_rejects_a_request_without_client_id():
    headers = {k: v for k, v in HEADERS.items() if k != "client-id"}
    body, _ = _ota_post(headers)
    assert body == {"success": False, "message": "request error."}
