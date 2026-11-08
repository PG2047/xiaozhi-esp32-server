"""按设备覆盖配置：单模块部署（没有智控台）时的"差异化配置"。

智控台（manager-api）能给每台设备配不同的模型和提示词；只跑 xiaozhi-server 一个模块时配置是全局的。这个模块补上这一块：
连接初始化时按请求头里的 device-id 读本地 YAML，把这台设备的覆盖项合并进这条连接的 config（connection.py 每条连接都是
深拷贝，互不影响）。每次连接都重读文件（几 KB），改完表让设备重连即生效，不用重启。

文件路径：config["device_overrides_file"]，默认 data/devices.yaml；文件不存在 = 不覆盖。schema v1：

```yaml
version: 1
devices:
  "aa:bb:cc:dd:ee:ff":          # device-id（设备 MAC，小写）
    name: 书房
    enabled: true               # false 时这里不处理（准入由 server.auth 或前置网关负责）
    brain: cascade              # cascade = 原版 VAD→ASR→LLM→TTS；填 S2S 配置块里的名字（OpenAIRealtime）或它的 type（openai_realtime）= 用该语音模型
    prompt: null                # 非空则替换 config["prompt"]
    llm: null                   # 非空且在 LLM 配置块里存在 → selected_module.LLM
    tts: null                   # 同上，TTS
    memory: true                # false = 这台设备的连接不带记忆、也不存记忆（跑批量测试前关掉，免得测试对话进日常记忆）
    s2s:                        # 按设备覆盖所选 S2S 配置块里的项（只对这条连接生效；brain 是级联时忽略）
      echo_gate: false          #   例：DSP 板自带 AEC 的设备关掉服务端回声门控（本地插话依赖门控判决，会一起关掉）
      vad_threshold: 0.6        #   例：电视旁边的设备把模型侧 VAD 门槛调高
```

任何错误都只记日志、不抛出：配置文件坏了不能让设备连不上。
"""
from __future__ import annotations

import os

import yaml

TAG = __name__
DEFAULT_FILE = "data/devices.yaml"


def load_device_overrides(path: str, device_id: str | None) -> dict | None:
    """返回这台设备的覆盖项字典；文件不存在、格式错、没登记、被停用都返回 None。"""
    if not device_id or not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    devices = data.get("devices") if isinstance(data, dict) else None
    if not isinstance(devices, dict):
        return None
    for key, info in devices.items():
        if str(key).lower() == str(device_id).lower():
            if not isinstance(info, dict) or info.get("enabled", True) is False:
                return None
            return info
    return None


def _resolve_s2s_name(s2s_block: dict, brain: str) -> str | None:
    """brain 既可以写 S2S 配置块里的名字（OpenAIRealtime），也可以写它的 type（openai_realtime）。"""
    if brain in s2s_block:
        return brain
    for name, cfg in s2s_block.items():
        if isinstance(cfg, dict) and str(cfg.get("type", name)) == brain:
            return name
    return None


def apply_device_overrides(conn) -> dict:
    """把这台设备的覆盖项合并进 conn.config；返回实际应用了哪些（给日志 / 测试用）。"""
    applied: dict = {}
    try:
        config = conn.config
        path = config.get("device_overrides_file", DEFAULT_FILE)
        info = load_device_overrides(path, getattr(conn, "device_id", None))
        if not info:
            return applied
        # 状态页台账用名字指代设备，不写 MAC（core/utils/status_ledger.py）；和 gate 的 /status 同一个规则
        conn.device_name = str(info.get("name") or f"未命名设备 {str(conn.device_id)[-2:]}")
        selected = config.setdefault("selected_module", {})

        brain = info.get("brain")
        if brain is not None:
            brain = str(brain).strip()
            if brain in ("", "cascade"):
                selected["S2S"] = ""
                applied["S2S"] = ""
            else:
                name = _resolve_s2s_name(config.get("S2S") or {}, brain)
                if name:
                    selected["S2S"] = name
                    applied["S2S"] = name
                else:
                    conn.logger.bind(tag=TAG).warning(f"设备覆盖的 brain={brain} 既不是 S2S 配置块的名字也不是其中某项的 type，忽略")

        prompt = info.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            config["prompt"] = prompt
            applied["prompt"] = True

        for key, section in (("llm", "LLM"), ("tts", "TTS")):
            name = info.get(key)
            if isinstance(name, str) and name.strip():
                if name in (config.get(section) or {}):
                    selected[section] = name
                    applied[section] = name
                else:
                    conn.logger.bind(tag=TAG).warning(f"设备覆盖的 {key}={name} 不在 {section} 配置块里，忽略")

        if info.get("memory") is False:
            conn.memory_disabled = True     # connection._initialize_memory 据此把这条连接的记忆置空
            applied["memory"] = False

        s2s_over = info.get("s2s")
        if isinstance(s2s_over, dict) and s2s_over:
            name = selected.get("S2S")
            block = (config.get("S2S") or {}).get(name) if name else None
            if isinstance(block, dict):
                for k, v in s2s_over.items():
                    if isinstance(v, (dict, list)):
                        conn.logger.bind(tag=TAG).warning(f"设备覆盖 s2s.{k} 只接受标量，忽略")
                        continue
                    block[str(k)] = v          # conn.config 是每条连接的深拷贝，改这里不影响别的连接
                applied["s2s"] = sorted(str(k) for k, v in s2s_over.items() if not isinstance(v, (dict, list)))
            else:
                conn.logger.bind(tag=TAG).warning("设备覆盖里有 s2s 项，但这台设备没有选 S2S 大脑（级联），忽略")
        elif s2s_over is not None:
            conn.logger.bind(tag=TAG).warning("设备覆盖的 s2s 必须是键值表，忽略")

        if applied:
            conn.logger.bind(tag=TAG).info(f"已应用设备覆盖: {applied}")
    except Exception as e:  # noqa: BLE001
        conn.logger.bind(tag=TAG).warning(f"应用设备覆盖失败，按全局配置继续: {type(e).__name__}: {e}")
    return applied
