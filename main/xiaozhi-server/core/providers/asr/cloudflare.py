"""Cloudflare Workers AI 语音识别（默认 @cf/openai/whisper-large-v3-turbo）。

Workers AI 没有 OpenAI 兼容的音频接口，所以不能复用 openai.py：POST .../ai/run/<模型>，
JSON 里放 base64 的 WAV，回 {"result": {"text": ...}, "success": true}。
失败（没额度、超时、回的不是这个结构）只记状态码和错误类型，返回空串，和别的 provider 一样按"没听到"处理。
"""
import base64
import os
import time
from typing import List, Optional, Tuple

import aiohttp

from config.logger import setup_logging
from core.providers.asr.base import ASRProviderBase
from core.providers.asr.dto.dto import InterfaceType

TAG = __name__
logger = setup_logging()

DEFAULT_BASE_URL = "https://api.cloudflare.com/client/v4"
DEFAULT_MODEL = "@cf/openai/whisper-large-v3-turbo"


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        self.interface_type = InterfaceType.NON_STREAM
        self.account_id = str(config.get("account_id") or "")
        self.api_token = str(config.get("api_token") or "")
        if not self.account_id or not self.api_token:
            raise ValueError("Cloudflare ASR 需要配置 account_id 和 api_token")
        self.base_url = str(config.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self.model = str(config.get("model_name") or DEFAULT_MODEL)
        self.language = str(config.get("language") or "zh")
        # 提示词要短：列词汇表（小智、灯、窗帘……）时安静的片段会幻听出"请不吝点赞 订阅"，词也会漏进结果
        # 不带提示词时偶尔出繁体。"以下是普通话。"这一句全部简体，也没有幻听（11-05 实测）
        self.initial_prompt = str(config.get("initial_prompt") or "")
        self.timeout_s = float(config.get("timeout_s", 15))
        self.output_dir = config.get("output_dir") or "tmp/"
        self.delete_audio_file = delete_audio_file
        os.makedirs(self.output_dir, exist_ok=True)

    def _body(self, wav: bytes) -> dict:
        body = {
            "audio": base64.b64encode(wav).decode("ascii"),
            "language": self.language,
            "vad_filter": True,
            "condition_on_previous_text": False,
        }
        if self.initial_prompt:
            body["initial_prompt"] = self.initial_prompt
        return body

    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, artifacts=None
    ) -> Tuple[Optional[str], Optional[str]]:
        if artifacts is None:
            return "", None
        file_path = artifacts.file_path
        wav = self._pcm_to_wav(artifacts.pcm_bytes)
        if not wav:
            return "", file_path
        url = f"{self.base_url}/accounts/{self.account_id}/ai/run/{self.model}"
        start = time.monotonic()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_s)) as s:
                async with s.post(url, json=self._body(wav),
                                  headers={"Authorization": f"Bearer {self.api_token}"}) as r:
                    status = r.status
                    data = await r.json(content_type=None) if status == 200 else None
        except Exception as e:  # noqa: BLE001  网络错误、超时、回的不是 JSON
            logger.bind(tag=TAG).error(f"Cloudflare 语音识别失败 {type(e).__name__}")
            return "", file_path
        if status != 200:
            logger.bind(tag=TAG).error(f"Cloudflare 语音识别失败 HTTP {status}")
            return "", file_path
        result = data.get("result") if isinstance(data, dict) else None
        text = result.get("text") if isinstance(result, dict) else None
        if not isinstance(text, str):
            logger.bind(tag=TAG).error("Cloudflare 语音识别失败：回复里没有 result.text")
            return "", file_path
        text = text.strip()
        # 不写转写内容（识别结果另有日志）。结果为空也要留一条：不然分不清是转写没听出来，还是服务端的 VAD 根本没切出这一句
        logger.bind(tag=TAG).info(f"Cloudflare 语音识别 {time.monotonic() - start:.2f} s，音频 {len(artifacts.pcm_bytes) / 32000:.1f} s"
                                  + ("" if text else "，结果为空"))
        return text, file_path
