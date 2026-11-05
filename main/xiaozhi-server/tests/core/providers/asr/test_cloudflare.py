"""Cloudflare Workers AI 转写：请求长什么样、回复怎么解析、失败时按"没听到"处理。上游是假的。"""
import asyncio
import base64
import io
import wave

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from core.providers.asr.base import ASRProviderBase
from core.providers.asr.cloudflare import ASRProvider
from core.utils import asr as asr_factory

PCM = b"\x10\x20" * 1600   # 100 ms @16k
_real_sleep = asyncio.tasks.sleep   # conftest 的 mock 模式把 asyncio.sleep 换成了不等待的版本


def _artifacts(pcm=PCM, file_path=None):
    return ASRProviderBase.AudioArtifacts(pcm_frames=[pcm], pcm_bytes=pcm, file_path=file_path, temp_path=None)


def _provider(base_url, tmp_path, **over):
    cfg = {"account_id": "acct", "api_token": "tok", "base_url": base_url, "output_dir": str(tmp_path), **over}
    return ASRProvider(cfg, True)


async def _upstream(reply, status=200, delay=0.0):
    """假的 Workers AI：记下请求路径、Authorization 和请求体。reply 是 dict 回 JSON，str 原样回。"""
    seen = []

    async def handler(request):
        seen.append({"path": request.path, "auth": request.headers.get("Authorization"), "body": await request.json()})
        if delay:
            await _real_sleep(delay)
        if isinstance(reply, dict):
            return web.json_response(reply, status=status)
        return web.Response(text=reply, status=status)
    app = web.Application()
    app.router.add_post("/{tail:.*}", handler)
    srv = TestServer(app)
    await srv.start_server()
    return srv, seen


def _run(reply, tmp_path, status=200, delay=0.0, artifacts=None, **over):
    async def go():
        srv, seen = await _upstream(reply, status, delay)
        try:
            p = _provider(str(srv.make_url("/client/v4")), tmp_path, **over)
            out = await p.speech_to_text([PCM], "s", artifacts if artifacts is not None else _artifacts())
        finally:
            await srv.close()
        return out, seen
    return asyncio.run(go())


def test_sends_wav_as_base64_to_the_model_path_and_strips_the_text(tmp_path):
    (text, path), seen = _run({"result": {"text": " 把书房灯关了 "}, "success": True}, tmp_path,
                              language="zh", initial_prompt="以下是普通话。")
    assert text == "把书房灯关了" and path is None
    req = seen[0]
    assert req["path"] == "/client/v4/accounts/acct/ai/run/@cf/openai/whisper-large-v3-turbo"
    assert req["auth"] == "Bearer tok"
    body = req["body"]
    assert body["language"] == "zh" and body["vad_filter"] is True
    assert body["initial_prompt"] == "以下是普通话。" and body["condition_on_previous_text"] is False
    with wave.open(io.BytesIO(base64.b64decode(body["audio"])), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.readframes(w.getnframes()) == PCM


def test_no_initial_prompt_key_when_not_configured(tmp_path):
    _, seen = _run({"result": {"text": "好"}, "success": True}, tmp_path)
    assert "initial_prompt" not in seen[0]["body"]


def test_returns_the_saved_file_path_when_there_is_one(tmp_path):
    (_, path), _seen = _run({"result": {"text": "好"}, "success": True}, tmp_path, artifacts=_artifacts(file_path="x.wav"))
    assert path == "x.wav"


@pytest.mark.parametrize("status,reply", [
    (429, {"success": False, "errors": [{"code": 3036, "message": "over quota"}]}),   # 免费额度用完
    (401, {"success": False, "errors": []}),
    (200, {"success": False, "result": None}),
    (200, {"result": {}}),
    (200, {"result": "不是对象"}),
    (200, ["不是对象"]),
    (200, "不是 JSON"),
])
def test_failures_count_as_nothing_heard(tmp_path, status, reply):
    if isinstance(reply, list):
        reply = '["不是对象"]'
    (text, path), seen = _run(reply, tmp_path, status=status)
    assert text == "" and path is None and len(seen) == 1


def test_timeout_counts_as_nothing_heard(tmp_path):
    (text, _), _seen = _run({"result": {"text": "太晚了"}}, tmp_path, delay=0.3, timeout_s=0.05)
    assert text == ""


def test_no_artifacts_or_no_audio_does_not_call_upstream(tmp_path):
    async def go():
        srv, seen = await _upstream({"result": {"text": "x"}})
        try:
            p = _provider(str(srv.make_url("/client/v4")), tmp_path)
            assert await p.speech_to_text([], "s", None) == ("", None)
            assert await p.speech_to_text([], "s", _artifacts(pcm=b"", file_path="y.wav")) == ("", "y.wav")
        finally:
            await srv.close()
        assert seen == []
    asyncio.run(go())


@pytest.mark.parametrize("over", [{"account_id": ""}, {"api_token": ""}])
def test_missing_credentials_refuse_to_start(tmp_path, over):
    cfg = {"account_id": "acct", "api_token": "tok", "output_dir": str(tmp_path), **over}
    with pytest.raises(ValueError, match="account_id"):
        ASRProvider(cfg, True)


def test_defaults_point_at_cloudflare(tmp_path):
    p = ASRProvider({"account_id": "acct", "api_token": "tok", "output_dir": str(tmp_path)}, True)
    assert p.base_url == "https://api.cloudflare.com/client/v4" and p.model == "@cf/openai/whisper-large-v3-turbo"
    assert p.language == "zh" and p.requires_file() is False


def test_factory_loads_it_by_type_name(tmp_path):
    p = asr_factory.create_instance("cloudflare", {"account_id": "a", "api_token": "t", "output_dir": str(tmp_path)}, True)
    assert isinstance(p, ASRProvider)
