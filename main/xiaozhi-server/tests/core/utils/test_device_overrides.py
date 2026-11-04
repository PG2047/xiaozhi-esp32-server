"""按设备覆盖（data/devices.yaml）：选大脑、提示词、LLM / TTS；没登记 / 停用 / 文件坏 / 名字不存在都不改配置、不抛异常。"""
from __future__ import annotations

import copy

from config.logger import setup_logging
from core.utils.device_overrides import apply_device_overrides, load_device_overrides

LOGGER = setup_logging()

REG = """
version: 1
devices:
  "AA:BB:CC:DD:EE:01": {name: s3, client_id: "x", brain: OpenAIRealtime, prompt: "你是书房助手"}
  "aa:bb:cc:dd:ee:02": {name: p, brain: cascade, llm: OtherLLM, tts: NoSuchTTS}
  "aa:bb:cc:dd:ee:03": {name: paused, enabled: false, brain: OpenAIRealtime}
  "aa:bb:cc:dd:ee:04": {name: bad-brain, brain: Nope}
  "aa:bb:cc:dd:ee:05": {name: by-type, brain: openai_realtime}
  "aa:bb:cc:dd:ee:06": {name: xvf, brain: openai_realtime, s2s: {echo_gate: false, vad_threshold: 0.6, nested: {a: 1}}}
  "aa:bb:cc:dd:ee:07": {name: cascade-with-s2s, brain: cascade, s2s: {echo_gate: false}}
  "aa:bb:cc:dd:ee:08": {name: bad-s2s, brain: openai_realtime, s2s: 5}
"""


class Conn:
    def __init__(self, device_id, path):
        self.device_id = device_id
        self.logger = LOGGER
        self.config = {
            "device_overrides_file": path,
            "prompt": "全局提示词",
            "selected_module": {"LLM": "MainLLM", "TTS": "EdgeTTS", "S2S": ""},
            "S2S": {"OpenAIRealtime": {"type": "openai_realtime"}},
            "LLM": {"MainLLM": {}, "OtherLLM": {}},
            "TTS": {"EdgeTTS": {}},
        }


def test_brain_prompt_and_case_insensitive_mac(tmp_path):
    p = tmp_path / "devices.yaml"
    p.write_text(REG, encoding="utf-8")
    c = Conn("aa:bb:cc:dd:ee:01", str(p))
    assert apply_device_overrides(c) == {"S2S": "OpenAIRealtime", "prompt": True}
    assert c.config["selected_module"]["S2S"] == "OpenAIRealtime" and c.config["prompt"] == "你是书房助手"


def test_brain_accepts_provider_type_name(tmp_path):
    p = tmp_path / "devices.yaml"
    p.write_text(REG, encoding="utf-8")
    c = Conn("aa:bb:cc:dd:ee:05", str(p))
    assert apply_device_overrides(c) == {"S2S": "OpenAIRealtime"}    # 写 type 也能解析到配置块的名字


def test_cascade_and_llm_override_with_unknown_tts_ignored(tmp_path):
    p = tmp_path / "devices.yaml"
    p.write_text(REG, encoding="utf-8")
    c = Conn("aa:bb:cc:dd:ee:02", str(p))
    c.config["selected_module"]["S2S"] = "OpenAIRealtime"  # 全局选了 S2S，这台设备要级联
    assert apply_device_overrides(c) == {"S2S": "", "LLM": "OtherLLM"}
    assert c.config["selected_module"] == {"LLM": "OtherLLM", "TTS": "EdgeTTS", "S2S": ""}
    assert c.config["prompt"] == "全局提示词"


def test_s2s_overrides_merge_into_selected_block_only(tmp_path):
    """s2s 键值表合并进这台设备所选的 S2S 配置块：标量覆盖、嵌套忽略；级联设备的 s2s 忽略；非键值表忽略。"""
    p = tmp_path / "devices.yaml"
    p.write_text(REG, encoding="utf-8")
    c = Conn("aa:bb:cc:dd:ee:06", str(p))
    assert apply_device_overrides(c) == {"S2S": "OpenAIRealtime", "s2s": ["echo_gate", "vad_threshold"]}
    blk = c.config["S2S"]["OpenAIRealtime"]
    assert blk["echo_gate"] is False and blk["vad_threshold"] == 0.6 and "nested" not in blk and blk["type"] == "openai_realtime"
    c2 = Conn("aa:bb:cc:dd:ee:07", str(p))
    c2.config["selected_module"]["S2S"] = "OpenAIRealtime"
    assert apply_device_overrides(c2) == {"S2S": ""}
    assert c2.config["S2S"]["OpenAIRealtime"] == {"type": "openai_realtime"}     # 级联设备：全局 S2S 块原样
    c3 = Conn("aa:bb:cc:dd:ee:08", str(p))
    assert apply_device_overrides(c3) == {"S2S": "OpenAIRealtime"}
    assert c3.config["S2S"]["OpenAIRealtime"] == {"type": "openai_realtime"}


def test_no_change_cases(tmp_path):
    p = tmp_path / "devices.yaml"
    p.write_text(REG, encoding="utf-8")
    for dev in ("aa:bb:cc:dd:ee:03", "aa:bb:cc:dd:ee:04", "ff:ff:ff:ff:ff:ff", None):
        c = Conn(dev, str(p))
        before = copy.deepcopy(c.config)
        assert apply_device_overrides(c) == {}
        assert c.config == before
    c = Conn("aa:bb:cc:dd:ee:01", str(tmp_path / "missing.yaml"))
    assert apply_device_overrides(c) == {}
    p.write_text("{broken", encoding="utf-8")
    assert load_device_overrides(str(p), "aa:bb:cc:dd:ee:01") is None
    p.write_text("devices: [1, 2]\n", encoding="utf-8")
    assert load_device_overrides(str(p), "aa:bb:cc:dd:ee:01") is None


def test_device_name_comes_from_registry(tmp_path):
    p = tmp_path / "devices.yaml"
    p.write_text(REG + '  "aa:bb:cc:dd:ee:09": {brain: cascade}\n', encoding="utf-8")
    named = Conn("aa:bb:cc:dd:ee:01", str(p))
    apply_device_overrides(named)
    assert named.device_name == "s3"
    unnamed = Conn("aa:bb:cc:dd:ee:09", str(p))
    apply_device_overrides(unnamed)
    assert unnamed.device_name == "未命名设备 09"
    unknown = Conn("aa:bb:cc:dd:ee:99", str(p))
    apply_device_overrides(unknown)
    assert not hasattr(unknown, "device_name")
