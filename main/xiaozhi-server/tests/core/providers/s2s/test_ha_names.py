"""ha_names：GetLiveContext 解析、调度用的名单、地点守卫、参数整理。地名和设备都是虚构的。
运行：在 main/xiaozhi-server 目录下 `pytest tests/core/providers/s2s -q`。"""
from __future__ import annotations

import json

from core.providers.s2s import ha_names as H

# HA GetLiveContext 的正文（yaml，名字和区域都用 ", " 连成一个字符串）。甲地的除醛净化器和客厅纱帘有不带地名的别名
LIVE_TEXT = """Live Context: An overview of the areas and the devices in this smart home:
- names: 甲地1号除醛净化器, 除醛净化器, 甲地除醛净化器
  domain: fan
  state: 'on'
  areas: 甲地楼下1号, 甲地楼下, 1号, 甲地1号, 12
- names: 甲地1号客厅纱帘, 客厅纱帘, 甲地客厅纱帘, 客厅纱帘  窗帘电机
  domain: cover
  state: open
  areas: 甲地楼下1号, 甲地楼下, 1号, 甲地1号, 12
  attributes:
    device_class: curtain
    current_position: 100
- names: 甲地温度
  domain: sensor
  state: '21'
  areas: 甲地楼下1号, 甲地楼下, 1号, 甲地1号, 12
  attributes:
    device_class: temperature
- names: 乙地风扇, 老家风扇
  domain: fan
  state: 'off'
  areas: 乙地老家, 乙地, 老家, 家
- names: 乙地台灯, 老家台灯
  domain: light
  state: 'off'
  areas: 乙地老家, 乙地, 老家, 家
- names: 乙地加湿器开启
  domain: scene
  state: unknown
"""
# server_mcp 执行器交回的是 str(CallToolResult)
LIVE_RESULT = ("meta=None content=[TextContent(type='text', text="
               f"{json.dumps({'success': True, 'result': LIVE_TEXT}, ensure_ascii=False)!r}, annotations=None, meta=None)] "
               "structuredContent=None isError=False")

_TARGET = {"name": {"type": "string"}, "area": {"type": "string"}, "floor": {"type": "string"},
           "domain": {"type": "array", "items": {"type": "string"}}}
HA_TOOLS = [
    {"type": "function", "name": "hass_set_state", "description": "控制清单里的设备",
     "parameters": {"type": "object", "properties": {"entity_id": {"type": "string"}, "state": {"type": "object"}}}},
    {"type": "function", "name": "fan__HassFanSetSpeed", "description": "Sets a fan's speed by percentage",
     "parameters": {"type": "object", "properties": dict(_TARGET, percentage={"type": "integer"}), "required": ["percentage"]}},
    {"type": "function", "name": "homeassistant__GetLiveContext", "description": "Provides real-time information",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "area": {"type": "string"}}}},
    {"type": "function", "name": "intent__HassTurnOff", "description": "Turns off/closes a device or entity",
     "parameters": {"type": "object", "properties": dict(
         _TARGET, device_class={"type": "array", "items": {"type": "string", "enum": ["curtain", "shade", "switch"]}})}},
    {"type": "function", "name": "assist_satellite__HassBroadcast", "description": "Broadcast a message",
     "parameters": {"type": "object", "properties": {"message": {"type": "string"}}}},
]


def entries():
    return H.parse_live_context(LIVE_RESULT)


def test_parse_live_context_from_mcp_result_repr():
    es = entries()
    assert [e["names"][0] for e in es] == ["甲地1号除醛净化器", "甲地1号客厅纱帘", "甲地温度", "乙地风扇", "乙地台灯", "乙地加湿器开启"]
    assert es[1]["names"] == ["甲地1号客厅纱帘", "客厅纱帘", "甲地客厅纱帘", "客厅纱帘  窗帘电机"]
    assert es[1]["domain"] == "cover" and es[1]["device_class"] == "curtain"
    assert es[0]["areas"] == ["甲地楼下1号", "甲地楼下", "1号", "甲地1号", "12"] and es[5]["areas"] == []
    assert H.parse_live_context(LIVE_TEXT) == es                     # 已经是正文也认
    assert H.parse_live_context("工具出错：timeout") == [] and H.parse_live_context("- names: [") == []


def test_dispatch_text_lists_controllable_devices_and_areas():
    t = H.dispatch_text(HA_TOOLS, entries())
    lines = t.split("\n")
    # 规则：清单里的设备照旧走 hass_*；HA 工具前缀从工具清单来，intent__ 排前面；不找设备的 HassBroadcast 不算
    assert lines[0].startswith("智能设备列表里有的设备照旧用 hass_set_state / hass_get_state。列表里没有的设备才用 intent__、fan__ 开头的")
    assert lines[1] == H.LIST_HEAD
    # 每台：第一个名字 + 最短的几个别名；跳过"设备名  实体名"；只列开关得动的 domain（不列传感器、灯、场景）
    assert lines[2:5] == ["- 甲地1号除醛净化器 / 除醛净化器 / 甲地除醛净化器", "- 甲地1号客厅纱帘 / 客厅纱帘 / 甲地客厅纱帘",
                          "- 乙地风扇 / 老家风扇"]
    assert "甲地温度" not in t and "乙地台灯" not in t and "开启" not in t
    assert lines[5] == "Home Assistant 的区域（area 只能填这些名字）：甲地楼下1号（也叫 甲地楼下、1号、甲地1号、12）；乙地老家（也叫 乙地、老家、家）"
    assert lines[6] == H.AMBIGUOUS and len(lines) == 7


def test_dispatch_text_rule_only_without_names_and_nothing_without_ha_intents():
    rule = H.dispatch_text(HA_TOOLS, [])
    assert "\n" not in rule and "不要把地点或房间拆到 area" in rule
    assert H.dispatch_text([HA_TOOLS[0], HA_TOOLS[2], HA_TOOLS[4]], entries()) == ""
    assert H.dispatch_text(HA_TOOLS[1:], []).startswith("控制设备时用 intent__、fan__ 开头的")   # 没有 hass_set_state 时不提它
    capped = H.dispatch_text(HA_TOOLS, entries(), max_entities=1)
    assert "- 甲地1号客厅纱帘" not in capped and "（还有 2 台没列出）" in capped


def test_place_guard_blocks_device_in_another_area():
    es = entries()
    msg = H.place_conflict("把乙地的除醛净化器关了", "intent__HassTurnOff", {"name": "除醛净化器"}, es)
    assert msg and msg.startswith("没有执行") and "乙地老家" in msg and "甲地楼下1号" in msg
    # 区域级：用户说乙地，模型填了甲地的区域
    assert H.place_conflict("把乙地的窗帘都关上", "intent__HassTurnOff", {"area": "甲地楼下1号", "domain": ["cover"]}, es)


def test_place_guard_lets_through_same_area_no_place_and_no_match():
    es = entries()
    off = "intent__HassTurnOff"
    assert H.place_conflict("把甲地的除醛净化器关了", off, {"name": "除醛净化器"}, es) is None    # "甲地"是只属于甲地的别名前缀
    assert H.place_conflict("关掉除醛净化器", off, {"name": "除醛净化器"}, es) is None            # 没说地点
    assert H.place_conflict("把乙地的窗帘打开", off, {"name": "乙地窗帘"}, es) is None            # HA 本来就找不到
    assert H.place_conflict("关掉乙地加湿器", off, {"name": "乙地加湿器开启"}, es) is None        # 命中的实体没分区域
    assert H.place_conflict("乙地的风扇开到最大", "fan__HassFanSetSpeed", {"area": "乙地", "percentage": 100}, es) is None
    # 单字、纯数字的区域别名不当地点词："家""12"在别的话里太常见
    assert H.place_conflict("家里的除醛净化器关了", off, {"name": "除醛净化器"}, es) is None
    assert H.mentioned_areas("调到12档", H.area_tokens(es)) == set()


def test_tidy_args_moves_device_class_out_of_domain():
    es, tool = entries(), HA_TOOLS[3]
    assert H.tidy_args(tool, {"area": "甲地楼下1号", "domain": ["curtain"]}, es) == {"area": "甲地楼下1号", "device_class": ["curtain"]}
    assert H.tidy_args(tool, {"domain": ["fan", "curtain"], "device_class": ["shade"]}, es) == {
        "domain": ["fan"], "device_class": ["shade", "curtain"]}
    assert H.tidy_args(tool, {"name": "乙地风扇", "domain": ["air purifier"]}, es) == {"name": "乙地风扇"}
    same = {"area": "乙地", "domain": ["fan"]}
    assert H.tidy_args(tool, same, es) is same and H.tidy_args(tool, {"domain": ["curtain"]}, []) == {"domain": ["curtain"]}
