from ..base import MemoryProviderBase, logger
import time
import json
import os
import threading
import yaml
from config.config_loader import get_project_dir
from config.manage_api_client import generate_and_save_chat_summary
import asyncio
from core.utils.util import check_model_key


short_term_memory_prompt = """
# 时空记忆编织者

## 核心使命
构建可生长的动态记忆网络，在有限空间内保留关键信息的同时，智能维护信息演变轨迹
根据对话记录，总结user的重要信息，以便在未来的对话中提供更个性化的服务

## 记忆法则
### 1. 三维度记忆评估（每次更新必执行）
| 维度       | 评估标准                  | 权重分 |
|------------|---------------------------|--------|
| 时效性     | 信息新鲜度（按对话轮次） | 40%    |
| 情感强度   | 含💖标记/重复提及次数     | 35%    |
| 关联密度   | 与其他信息的连接数量      | 25%    |

### 2. 动态更新机制
**名字变更处理示例：**
原始记忆："曾用名": ["张三"], "现用名": "张三丰"
触发条件：当检测到「我叫X」「称呼我Y」等命名信号时
操作流程：
1. 将旧名移入"曾用名"列表
2. 记录命名时间轴："2024-02-15 14:32:启用张三丰"
3. 在记忆立方追加：「从张三到张三丰的身份蜕变」

### 3. 空间优化策略
- **信息压缩术**：用符号体系提升密度
  - ✅"张三丰[北/软工/🐱]"
  - ❌"北京软件工程师，养猫"
- **淘汰预警**：当总字数≥900时触发
  1. 删除权重分<60且3轮未提及的信息
  2. 合并相似条目（保留时间戳最近的）

## 记忆结构
输出格式必须为可解析的json字符串，不需要解释、注释和说明，保存记忆时仅从对话提取信息，不要混入示例内容
```json
{
  "时空档案": {
    "身份图谱": {
      "现用名": "",
      "特征标记": [] 
    },
    "记忆立方": [
      {
        "事件": "入职新公司",
        "时间戳": "2024-03-20",
        "情感值": 0.9,
        "关联项": ["下午茶"],
        "保鲜期": 30 
      }
    ]
  },
  "关系网络": {
    "高频话题": {"职场": 12},
    "暗线联系": [""]
  },
  "待响应": {
    "紧急事项": ["需立即处理的任务"], 
    "潜在关怀": ["可主动提供的帮助"]
  },
  "高光语录": [
    "最打动人心的瞬间，强烈的情感表达，user的原话"
  ]
}
```
"""


def extract_json_data(json_code):
    start = json_code.find("```json")
    # 从start开始找到下一个```结束
    end = json_code.find("```", start + 1)
    # print("start:", start, "end:", end)
    if start == -1 or end == -1:
        try:
            jsonData = json.loads(json_code)
            return json_code
        except Exception as e:
            print("Error:", e)
        return ""
    jsonData = json_code[start + 7 : end]
    return jsonData


TAG = __name__

# 所有设备的记忆存在同一个 YAML 里，写（读-改-写）必须串行：不然两台设备同时断开时，后写的会把先写的那台冲掉。
# 可重入锁：save_memory 持锁期间会调 save_memory_to_file，后者单独被调用时也要加锁。
# 只读不加锁：写是临时文件 + 整体替换，读到的总是完整的旧文件或新文件。新连接在建会话之前就要读记忆，
# 而 save_memory 会持锁等几秒的总结请求，读也排队的话，刚断开又唤醒（或另一台设备这时连上）就要多等这几秒。
_FILE_LOCK = threading.RLock()


class MemoryProvider(MemoryProviderBase):
    def __init__(self, config, summary_memory):
        super().__init__(config)
        self.short_memory = ""
        self.save_to_file = True
        # 总结用的系统提示词：配置里写了 summary_prompt 就用它，不写用上面的默认提示词。输出必须是 JSON（保存前会校验）
        self.summary_prompt = str((config or {}).get("summary_prompt") or "").strip() or short_term_memory_prompt
        self.memory_path = get_project_dir() + "data/.memory.yaml"
        self.load_memory(summary_memory)

    def init_memory(
        self, role_id, llm, summary_memory=None, save_to_file=True, **kwargs
    ):
        super().init_memory(role_id, llm, **kwargs)
        self.save_to_file = save_to_file
        self.load_memory(summary_memory)

    def load_memory(self, summary_memory):
        # api获取到总结记忆后直接返回
        if summary_memory or not self.save_to_file:
            self.short_memory = summary_memory
            return

        all_memory = self._read_all()   # 不加锁，见 _FILE_LOCK 的注释
        # 文件里没有这台设备就从空开始，不沿用这个实例之前装过的记忆
        self.short_memory = all_memory.get(self.role_id) or "" if self.role_id else ""

    def _read_all(self):
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def save_memory_to_file(self):
        with _FILE_LOCK:
            all_memory = self._read_all()
            all_memory[self.role_id] = self.short_memory
            # 先写临时文件再整体替换：写到一半进程被杀时，原文件还是完整的
            tmp_path = f"{self.memory_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.dump(all_memory, f, allow_unicode=True)
            os.replace(tmp_path, self.memory_path)

    async def save_memory(self, msgs, session_id=None):
        # 打印使用的模型信息
        model_info = getattr(self.llm, "model_name", str(self.llm.__class__.__name__))
        logger.bind(tag=TAG).debug(f"使用记忆保存模型: {model_info}")
        if self.llm is None:
            logger.bind(tag=TAG).error("LLM is not set for memory provider")
            return None
        memory_key_msg = check_model_key("记忆总结专用LLM", str(getattr(self.llm, "api_key", "") or ""))
        if memory_key_msg:
            logger.bind(tag=TAG).error(memory_key_msg)

        if len(msgs) < 2:
            return None

        if not self.save_to_file:
            # 当save_to_file为False时，调用Java端的聊天记录总结接口
            summary_id = session_id if session_id else self.role_id
            await generate_and_save_chat_summary(summary_id)
            logger.bind(tag=TAG).info("Save memory successful - chat summary requested")
            return self.short_memory

        # 从重读文件、调总结到写回一直持有锁（中间是几秒的网络请求，中途放锁重读就白做了）：
        # 同一台设备两段会话挨得很近、上一次总结还没写完时，这一次以文件里最新的记忆为"历史记忆"，不会把上一次的更新冲掉。
        # 总结都在连接关闭后的后台线程里跑，一天几十次，串行没有代价。
        with _FILE_LOCK:
            latest = self._read_all().get(self.role_id)
            if latest:
                self.short_memory = latest
            msgStr = self._summary_input(msgs)
            try:
                result = self.llm.response_no_stream(
                    self.summary_prompt,
                    msgStr,
                    max_tokens=2000,
                    temperature=0.2,
                )
                json_str = extract_json_data(result)
                json.loads(json_str)  # 检查json格式是否正确
                self.short_memory = json_str
                self.save_memory_to_file()
            except Exception as e:
                logger.bind(tag=TAG).error(f"Error in saving memory: {e}")
                return self.short_memory
        # 不记 role_id：它是设备 MAC，日志里不出现设备身份
        logger.bind(tag=TAG).info(f"Save memory successful - {len(self.short_memory or '')} chars")

        return self.short_memory

    def _summary_input(self, msgs) -> str:
        msgStr = ""
        for msg in msgs:
            content = msg.content

            # Extract content from JSON format if present (for ASR with emotion/language tags)
            try:
                if content and content.strip().startswith("{") and content.strip().endswith("}"):
                    data = json.loads(content)
                    if "content" in data:
                        content = data["content"]
            except (json.JSONDecodeError, KeyError, TypeError):
                # If parsing fails, use original content
                pass

            if msg.role == "user":
                msgStr += f"User: {content}\n"
            elif msg.role == "assistant":
                msgStr += f"Assistant: {content}\n"
        if self.short_memory and len(self.short_memory) > 0:
            msgStr += "历史记忆：\n"
            msgStr += self.short_memory

        # 当前时间
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        msgStr += f"当前时间：{time_str}"
        return msgStr

    async def query_memory(self, query: str) -> str:
        return self.short_memory
