"""ha_names：GetLiveContext 解析、调度用的名单、地点守卫、歧义检查、参数整理。地名和设备都是虚构的。
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


def test_dispatch_text_query_rule_follows_get_live_context_tool():
    t = H.dispatch_text(HA_TOOLS, entries())
    assert "查这些设备的状态用 homeassistant__GetLiveContext：name 也照这样填完整的设备名" in t.split("\n")[0]
    no_query = [x for x in HA_TOOLS if not x["name"].endswith("GetLiveContext")]
    assert "GetLiveContext" not in H.dispatch_text(no_query, entries())


def _area_entries():
    """丙地两台 fan（风扇、净化器）加一个传感器；丁地一台 fan 加一个传感器。"""
    return [
        {"names": ["丙地风扇", "丙地大风扇"], "domain": "fan", "areas": ["丙地家", "丙地"], "device_class": None},
        {"names": ["丙地净化器"], "domain": "fan", "areas": ["丙地家", "丙地"], "device_class": None},
        {"names": ["丙地温度"], "domain": "sensor", "areas": ["丙地家", "丙地"], "device_class": "temperature"},
        {"names": ["丁地风扇"], "domain": "fan", "areas": ["丁地家", "丁地"], "device_class": None},
        {"names": ["丁地温度"], "domain": "sensor", "areas": ["丁地家", "丁地"], "device_class": "temperature"},
    ]


def test_area_ambiguity_asks_which_one_when_several_match_without_all_words():
    es, speed, off = _area_entries(), "fan__HassFanSetSpeed", "intent__HassTurnOff"
    msg = H.area_ambiguity("丙地的风扇风速调到最大", speed, {"area": "丙地", "percentage": 100}, es)
    assert msg and msg.startswith("没有执行") and "2 台" in msg and "丙地风扇" in msg and "丙地净化器" in msg
    # 设备名被转写错（"除甲醛"→"出价全"）时的"全"不算"全部"
    assert H.area_ambiguity("把丙地的出价全净化器关了", off, {"area": "丙地", "domain": ["fan"]}, es)
    # 没填区域、只填类别：全屋的 fan
    assert "家里有 3 台" in H.area_ambiguity("关风扇", off, {"domain": ["fan"]}, es)


def test_area_ambiguity_lets_through_all_words_single_device_and_named_calls():
    es, speed, off = _area_entries(), "fan__HassFanSetSpeed", "intent__HassTurnOff"
    for said in ("把丙地的风扇都调到最大", "丙地的风扇全部调到最大", "所有风扇调到最大", "丙地的风扇全开"):
        assert H.area_ambiguity(said, speed, {"area": "丙地", "percentage": 100}, es) is None, said
    assert H.area_ambiguity("丁地的风扇关了", off, {"area": "丁地", "domain": ["fan"]}, es) is None       # 只有一台
    assert H.area_ambiguity("丁地的关了", off, {"area": "丁地"}, es) is None                              # 传感器不算可控设备
    assert H.area_ambiguity("丙地风扇关了", off, {"name": "丙地风扇", "area": "丙地"}, es) is None         # 填了 name
    assert H.area_ambiguity("把乙地的风扇关了", off, {"area": "没有的区域"}, es) is None                   # HA 本来就找不到


def test_mixed_kinds_only_for_on_off_calls_without_a_kind_or_name():
    es = entries()
    off, on = "intent__HassTurnOff", "intent__HassTurnOn"
    # 甲地有风扇、窗帘和传感器（传感器不算），乙地有风扇和灯；area 填别名也一样
    assert [e["domain"] for e in H.mixed_kinds(off, {"area": "甲地1号"}, es)] == ["fan", "cover"]
    assert [e["names"][0] for e in H.mixed_kinds(on, {"area": "乙地老家"}, es)] == ["乙地风扇", "乙地台灯"]
    # 没填 name 的几种写法：HA 看到的都是只有区域
    for args in ({"area": "乙地", "name": "all"}, {"area": "乙地", "name": "  "}, {"area": "乙地", "domain": None},
                 {"area": "乙地", "device_class": []}):
        assert len(H.mixed_kinds(off, args, es)) == 2
    # 填了类别或名字的不管：把甲地的窗帘都关上、把灯和风扇都关了（两类是用户说的）
    for args in ({"area": "甲地1号", "domain": ["cover"]}, {"area": "甲地1号", "device_class": ["curtain"]},
                 {"area": "乙地", "domain": ["light", "fan"]}, {"name": "乙地风扇"}, {"area": "乙地", "name": "乙地风扇"}):
        assert H.mixed_kinds(off, args, es) == []
    # 只有开关意图不限 domain：SetPosition / StopMoving 碰到窗帘以外的整个失败，灯和风扇的意图只找自己那一类
    assert H.mixed_kinds("intent__HassSetPosition", {"area": "甲地1号", "position": 0}, es) == []
    assert H.mixed_kinds("intent__HassStopMoving", {"area": "甲地1号"}, es) == []
    assert H.mixed_kinds("light__HassLightSet", {"area": "乙地"}, es) == []
    assert H.mixed_kinds("fan__HassFanSetSpeed", {"area": "乙地", "percentage": 50}, es) == []
    assert H.mixed_kinds(off, {"area": "没有的区域"}, es) == []


def test_mixed_kinds_ignores_entities_that_cannot_be_switched():
    """一类能开关的设备 + 没有开关服务的（档位、目标湿度、传感器）：HA 只动得了那一类，不用问。"""
    place = ["丙地"]
    es = [{"names": [n], "domain": d, "areas": place, "device_class": None}
          for n, d in (("丙地灯", "light"), ("丙地档位", "select"), ("丙地目标湿度", "number"), ("丙地温度", "sensor"))]
    assert H.mixed_kinds("intent__HassTurnOff", {"area": "丙地"}, es) == []
    assert [e["names"][0] for e in H.would_switch("intent__HassTurnOff", {"area": "丙地"}, es)] == ["丙地灯"]
    es.append({"names": ["丙地自定义"], "domain": "input_boolean", "areas": place, "device_class": None})   # 不认识的 domain 算能开关
    assert [e["domain"] for e in H.mixed_kinds("intent__HassTurnOff", {"area": "丙地"}, es)] == ["light", "input_boolean"]


def test_mixed_kinds_trusts_a_filled_device_class_even_across_domains():
    """device_class 跨了两类（窗帘 + 插座）也是填了类别的，照用户说的。"""
    es = [{"names": ["丙地窗帘"], "domain": "cover", "areas": ["丙地"], "device_class": "curtain"},
          {"names": ["丙地插座"], "domain": "switch", "areas": ["丙地"], "device_class": "outlet"}]
    assert H.mixed_kinds("intent__HassTurnOff", {"area": "丙地", "device_class": ["curtain", "outlet"]}, es) == []
    assert len(H.mixed_kinds("intent__HassTurnOff", {"area": "丙地"}, es)) == 2


def test_ask_kind_counts_each_kind_and_is_short_enough_to_hand_back():
    es = _name_entries()
    hits = H.mixed_kinds("intent__HassTurnOff", {"area": "戊地1号"}, es)
    text = H.ask_kind(hits, {"area": "戊地1号"})
    assert text.startswith("没有执行") and "戊地1号所有能开关的设备（窗帘 4 个、风扇或净化器 2 个、灯 1 个）" in text
    assert "问用户要控制哪一类" in text and "全部" in text and len(text) <= 200       # 超过 200 字就不直接交回 Live 了
    assert "家里" in H.ask_kind(hits, {})
    assert "input_boolean 1 个" in H.ask_kind(hits + [{"domain": "input_boolean"}], {})


def test_said_all_kinds_reads_only_the_last_sentence_and_not_a_named_kind():
    assert H.said_all_kinds("把甲地的都关了\n都关") and H.said_all_kinds("全部") and H.said_all_kinds("把甲地的都关了\n甲地的都关")
    assert not H.said_all_kinds("把甲地的都关了\n嗯") and not H.said_all_kinds("把甲地的都关了\n灯") and not H.said_all_kinds("")
    assert not H.said_all_kinds("窗帘都关了") and not H.said_all_kinds("所有的灯") and not H.said_all_kinds("净化器全关")


def _name_entries():
    """戊地四幅窗帘（卧室 / 客厅 × 布帘 / 纱帘，布帘也叫遮光窗帘、纱帘也叫薄窗帘）、一台风扇、一台净化器、一盏灯；
    己地一台同款风扇、一盏灯。别名里有 HA 拼的"设备名  实体名"。"""
    here, there = ["戊地楼下1号", "戊地楼下", "1号", "戊地1号"], ["己地老家", "己地", "老家"]

    def curtain(room, kind, other):
        return {"names": [f"戊地1号{room}{kind}", f"1号{room}{kind}  窗帘电机", f"{room}{kind}", f"{room}{other}",
                          f"戊地{room}{kind}", f"戊地{room}{other}"], "domain": "cover", "areas": here, "device_class": "curtain"}
    return [curtain("卧室", "布帘", "遮光窗帘"), curtain("卧室", "纱帘", "薄窗帘"),
            curtain("客厅", "布帘", "遮光窗帘"), curtain("客厅", "纱帘", "薄窗帘"),
            {"names": ["戊地108甲牌净化风扇", "戊地甲牌风扇"], "domain": "fan", "areas": here, "device_class": None},
            {"names": ["戊地108除醛净化器", "大净化器", "戊地除醛净化器"], "domain": "fan", "areas": here, "device_class": None},
            {"names": ["己地甲牌净化风扇", "己地甲牌风扇"], "domain": "fan", "areas": there, "device_class": None},
            {"names": ["戊地书房灯", "书房灯"], "domain": "light", "areas": here, "device_class": None},
            {"names": ["己地书房灯"], "domain": "light", "areas": there, "device_class": None}]


def _rivals(said, name, tool="intent__HassTurnOff", recent=None):
    group = H.name_rivals(said, said if recent is None else recent, tool, {"name": name}, _name_entries())
    return [e["names"][0] for e in group]


def test_name_rivals_lists_devices_the_user_words_do_not_tell_apart():
    curtains = ["戊地1号卧室布帘", "戊地1号卧室纱帘", "戊地1号客厅布帘", "戊地1号客厅纱帘"]
    fans = ["戊地108甲牌净化风扇", "己地甲牌净化风扇"]
    assert _rivals("把戊地的窗帘关上", "戊地1号卧室布帘") == curtains
    # 说了一半：只剩分不开的那几台
    assert _rivals("把戊地卧室的窗帘关上", "卧室布帘") == curtains[:2]
    assert _rivals("把戊地的纱帘关上", "戊地1号客厅纱帘") == [curtains[1], curtains[3]]
    # 没说地点：别处的同款也算；同处的净化器靠"风扇"分开了
    assert _rivals("打开甲牌风扇", "戊地甲牌风扇") == fans
    # 小模型挑错了：用户说纱帘，它填了布帘
    assert _rivals("把客厅纱帘关上", "客厅布帘") == [curtains[2], curtains[3]]
    # 别名里的"108"不因为"调到100"就算说清了
    assert _rivals("把风扇调到100", "戊地108甲牌净化风扇", "fan__HassFanSetSpeed") == fans
    # 同类只算同 domain：什么类别词都没说时，也不把风扇、灯列进来
    assert _rivals("把戊地的关了", "戊地1号卧室布帘") == curtains
    # HA 拼的"设备名  实体名"不算名字：只有它带"空气"，也不能靠这个词分开
    two = [{"names": ["甲净化器", "型号X  空气净化器"], "domain": "fan", "areas": [], "device_class": None},
           {"names": ["乙净化器"], "domain": "fan", "areas": [], "device_class": None}]
    assert H.name_rivals("把空气净化器关了", "把空气净化器关了", "intent__HassTurnOff", {"name": "甲净化器"}, two) == two


def test_says_all():
    for said in ("都关了吧", "把甲地的灯全部打开", "所有的灯", "灯全关了", "全打开", "一起关了", "统统关掉", "灯全灭了", "全屋的灯"):
        assert H.says_all(said), said
    # "全"后面不是动作的不算："除甲醛"曾被转写成"出价全"
    for said in ("把灯关了", "出价全净化器关了", "", None):
        assert not H.says_all(said), said


_LIGHTS = ("下面是我家智能设备列表（位置，设备名，entity_id），可以通过homeassistant控制\n"
           "甲地书房,书房灯,light.a1\n甲地1号,小台灯,light.a2\n甲地1号,音箱,media_player.a3\n"
           "老家卧室,卧室灯,light.b1\n丙地,吊灯,light.c1\n")


def _at_place(said: str, recent: str | None = None, entity_id: str = "light.a1", es: list | None = None, asked: bool = False) -> list[str]:
    targets, question = H.all_at_place(_LIGHTS, said, said if recent is None else recent, entity_id, entries() if es is None else es, asked)
    assert question is None
    return targets


def test_all_at_place_lists_same_kind_devices_in_the_named_place():
    # 位置写法不统一也算（"甲地书房""甲地1号"）；只要同类（音箱不算）；别处不算；第一盏填错也不影响
    assert _at_place("把甲地的灯全关了", entity_id="light.b1") == ["light.a1", "light.a2"]
    assert _at_place("把甲地的喇叭都关了", entity_id="media_player.a3") == ["media_player.a3"]
    # 区域的别名、两处地点
    assert _at_place("把老家的灯都关了") == ["light.b1"]
    assert _at_place("把甲地和老家的灯都关了") == ["light.a1", "light.a2", "light.b1"]


def test_all_at_place_takes_the_latest_sentence_with_a_place():
    assert _at_place("都关了吧", "甲地有哪些灯\n都关了吧") == ["light.a1", "light.a2"]
    # 这句说了地点：前几句提过别处也不算；这句没说，取最近提到的那一处
    assert _at_place("把老家的灯都打开", "甲地有哪些灯\n把老家的灯都打开") == ["light.b1"]
    assert _at_place("都关了吧", "甲地有哪些灯\n老家呢\n都关了吧") == ["light.b1"]


def test_all_at_place_gives_up_when_unsure():
    assert _at_place("都这么晚了，把甲地的书房灯关了") == []             # 点了具体设备
    assert _at_place("把甲地的灯都打开，除了台灯") == []
    # 排除的词：地点可能正是要排除的
    for said in ("除了甲地，灯都关了", "把老家的灯都关了，甲地的不用", "甲地的灯留一盏，都关了", "别把甲地的灯都关了", "不要把甲地的灯都关了",
                 "甲地的灯都不关了", "甲地的灯都算了", "甲地的灯都关了吗"):
        assert _at_place(said) == [], said
    # 取不到名单（认不出地点）、设备列表为空，都不展开；表头那行不算设备
    assert _at_place("把甲地的灯全关了", es=[]) == []
    assert H.all_at_place("你是小智", "把甲地的灯全关了", "把甲地的灯全关了", "light.a1", entries()) == ([], None)


def test_all_at_place_asks_where_without_a_place():
    targets, question = H.all_at_place(_LIGHTS, "把灯全关了", "把灯全关了", "light.a1", entries())
    assert targets == [] and question.startswith("没有执行") and "甲地楼下1号、乙地老家" in question and "一共 4 台" in question
    # 这类设备只在一处：不用问；名单没取到（认不出哪一处）：不问也不展开
    assert _at_place("把喇叭都关了", entity_id="media_player.a3") == ["media_player.a3"]
    assert _at_place("把灯全关了", es=[]) == []


def test_all_at_place_after_asking_where():
    def answer(said):
        return _at_place(said, f"把灯全关了\n{said}", asked=True)
    assert answer("甲地的") == ["light.a1", "light.a2"]
    assert answer("甲地和老家") == ["light.a1", "light.a2", "light.b1"]
    assert answer("都关了吧") == ["light.a1", "light.a2", "light.b1", "light.c1"]    # 全部：位置对不上区域的也算
    for said in ("书房灯", "不用了", "都不是", "都不关"):
        assert answer(said) == [], said
    # 没答上来、还没听到这一句：再问一遍
    for said, recent in (("嗯", "把灯全关了\n嗯"), ("第一个", "把灯全关了\n第一个"), ("", "把灯全关了")):
        targets, question = H.all_at_place(_LIGHTS, said, recent, "light.a1", entries(), True)
        assert targets == [] and question.startswith("没有执行"), said


def test_device_rows_only_takes_lines_ending_in_an_entity_id():
    prompt = "你是小智，丁地,戊地,己地都有\n甲地书房, 书房灯, light.a1\n两栏,light.a2\n四栏,一,二,light.a3\n"
    assert H._device_rows(prompt) == [["甲地书房", "书房灯", "light.a1"]]
    assert H._device_rows("") == [] and H._device_rows(None) == []


def _listed_conflict(said: str, entity_id: str = "light.a1", prompt: str = _LIGHTS, es: list | None = None) -> str | None:
    return H.listed_place_conflict(prompt, said, entity_id, entries() if es is None else es)


def test_listed_place_conflict_blocks_a_device_elsewhere():
    text = _listed_conflict("把老家的灯也都关了")
    assert text.startswith("没有执行") and "乙地老家" in text and "甲地书房" in text
    # 位置写法不统一也认得出（"甲地1号"）；别的类也算（音箱）；反过来也一样；地点的别名也算
    assert _listed_conflict("把老家的灯也都关了", "light.a2")
    assert _listed_conflict("把老家的都关了", "media_player.a3")
    assert _listed_conflict("把甲地的灯也都关了", "light.b1")
    assert _listed_conflict("乙地的灯也都关了", "light.a1")


def test_listed_place_conflict_lets_through_the_place_said_and_unsure_cases():
    # 就在说的地方，或者这句提到了两处
    for said, entity in (("把甲地的灯都关了", "light.a1"), ("把甲地的灯都关了", "light.a2"), ("把老家的灯都关了", "light.b1"),
                         ("把甲地和老家的灯都关了", "light.a1"), ("把甲地和老家的灯都关了", "light.b1")):
        assert _listed_conflict(said, entity) is None, said
    # 这句没提地点（"都关了吧"看前一句），还没听到这一句
    for said in ("都关了吧", "把它打开", "把灯关了", "", None):
        assert _listed_conflict(said) is None, said
    # 地点只看带动作的小句：说老家的是天气；一个带动作的小句都没有才看整句
    assert _listed_conflict("老家今天多少度，顺便把书房灯关了") is None
    assert _listed_conflict("先把书房灯关了，老家今天多少度") is None
    assert _listed_conflict("老家今天多少度，顺便把老家的灯关了") and _listed_conflict("书房灯关了，再把老家的灯关了")
    assert _listed_conflict("那老家的呢") and _listed_conflict("老家的")
    # 排除、否定、提问：提到的地点可能正好不是要控制的
    for said in ("除了老家，把灯都关了", "老家的灯留着，别的都关了", "老家的不用关", "别管老家", "老家的灯算了", "老家的灯都关了吗"):
        assert _listed_conflict(said) is None, said
    # 设备不在列表里；位置里认不出地点（"丙地"不是 HA 的区域）；列表或名单为空
    assert _listed_conflict("把老家的灯关了", "light.unknown") is None
    assert _listed_conflict("把老家的灯关了", "") is None
    assert _listed_conflict("把老家的灯关了", "light.c1") is None
    assert _listed_conflict("把老家的灯关了", prompt="你是小智") is None
    assert _listed_conflict("把老家的灯关了", es=[]) is None


_HERE, _HOME = "甲地楼下1号", "乙地老家"


def _excluded(said: str | None) -> set[str]:
    return H.excluded_places(said, H.area_tokens(entries()))


def test_excluded_places_recognises_the_usual_ways_of_saying_it():
    for said in ("除了甲地，把灯都关了", "除了甲地都关了", "除甲地外都关了", "除了甲地以外的灯都关了", "除开甲地，别的都关了", "除去甲地，其他的都关了",
                 "除了甲地的灯，别的都关了", "除了甲地那边，都关了", "除了甲地把灯都关了", "把灯都关了，除了甲地",
                 "把灯都关了，甲地的不用", "把灯都关了，甲地不用", "把灯都关了甲地的不用", "甲地的灯不用关，别的都关了", "甲地的灯都不用关了", "甲地的先不用",
                 "甲地的留着", "甲地的灯留着", "甲地除外", "甲地的算了", "甲地别动", "甲地的先别关", "甲地的灯不关",
                 "别关甲地的灯", "不要关甲地的灯", "不关甲地的", "不用管甲地", "先别动甲地的", "别把甲地的灯都关了", "不要把甲地的灯都关了", "留着甲地的灯", "不包括甲地", "不含甲地"):
        assert _excluded(said) == {_HERE}, said
    assert _excluded("除了甲地和老家，把灯都关了") == {_HERE, _HOME}
    assert _excluded("把甲地的灯都关了，老家的不用") == {_HOME}
    assert _excluded("除了甲地，老家的也不用") == {_HERE, _HOME}
    # 上次委托以来的几句话：换行也是小句的边界
    assert _excluded("除了甲地\n都关了吧") == {_HERE}


def test_excluded_places_leaves_everything_else_alone():
    for said in ("把灯都关了", "把甲地的灯都关了", "把甲地和老家的灯都关了", "那老家的呢",
                 "除了甲地书房灯，把灯都关了", "除了书房灯，把甲地的灯都关了", "把甲地的灯都打开，除了台灯",     # 排除的是一盏灯
                 "把除醛净化器关了", "把甲地的除醛净化器关了", "甲地除醛净化器关了",                              # 设备名里的"除"
                 "除了关甲地的灯，还要把老家的灯打开",                                                            # "除了"后面不是地点
                 "要不把甲地的灯关了", "要不要把甲地的灯关了", "用不用把甲地的灯关了", "能不能把甲地的灯关了", "不如把甲地的灯关了",     # 在提议
                 "甲地的灯不亮，把老家的灯关了", "甲地的灯都关了吗", "甲地的别的灯都关了", "甲地的灯留一盏，都关了", "", None):
        assert _excluded(said) == set(), said
    for said in ("除了甲地，把灯都关了", "甲地的不用", "别关甲地的灯"):     # 没取到名单
        assert H.excluded_places(said, {}) == set(), said


def test_listed_place_conflict_reads_an_excluded_place_the_other_way_round():
    text = _listed_conflict("除了老家，把灯都关了", "light.b1")
    assert text.startswith("没有执行") and "不动乙地老家" in text and "老家卧室" in text
    assert _listed_conflict("除了老家，把灯都关了", "light.a1") is None
    # 句子后半截才说不动哪里；不动的地方带动作词
    assert _listed_conflict("把灯都关了，甲地的不用", "light.a1") and _listed_conflict("把灯都关了，甲地的不用", "light.a2")
    assert _listed_conflict("把灯都关了，甲地的不用", "light.b1") is None
    assert _listed_conflict("甲地的灯不用关，别的都关了", "light.a1") and _listed_conflict("甲地的灯不用关，别的都关了", "light.b1") is None
    # 说了要控制的和不动的：两边都看；认不出地点的设备放行
    assert _listed_conflict("把老家的灯都关了，甲地的不用", "light.a1") and _listed_conflict("把老家的灯都关了，甲地的不用", "light.b1") is None
    assert _listed_conflict("把老家的灯都关了，甲地的不用", "light.c1") is None
    assert _listed_conflict("除了老家，把灯都关了", "light.unknown") is None and _listed_conflict("除了老家，把灯都关了", "light.c1") is None
    # 认不出是不动哪里的：还是放行
    assert _listed_conflict("除了老家卧室灯，把灯都关了", "light.b1") is None
    assert _listed_conflict("老家的灯留一盏，都关了", "light.a1") is None and _listed_conflict("老家的灯留一盏，都关了", "light.b1") is None


def test_place_conflict_reads_an_excluded_place_the_other_way_round():
    es, off = entries(), "intent__HassTurnOff"
    # 只说了不动哪里：别处放行（之前"甲地"被当成要控制的地方，别处的调用全拦了），那里的不执行
    assert H.place_conflict("除了甲地，把净化器都关了", off, {"name": "乙地风扇"}, es) is None
    msg = H.place_conflict("除了甲地，把净化器都关了", off, {"name": "除醛净化器"}, es)
    assert msg.startswith("没有执行") and "不动甲地楼下1号" in msg
    # 只填类别、会一起动到不动的那里：不执行；填了别处的区域放行
    assert H.place_conflict("把风扇都关了，甲地的不用", off, {"domain": ["fan"]}, es)
    assert H.place_conflict("把风扇都关了，甲地的不用", off, {"area": "乙地", "domain": ["fan"]}, es) is None
    # 说了要控制的和不动的
    assert H.place_conflict("把乙地的风扇关了，甲地的不用", off, {"name": "除醛净化器"}, es)
    assert H.place_conflict("把乙地的风扇关了，甲地的不用", off, {"name": "乙地风扇"}, es) is None
    # 排除的是一台设备：认不出是不动哪里，甲地还算提到过，和以前一样
    assert H.place_conflict("除了甲地除醛净化器，把乙地的都关了", off, {"name": "除醛净化器"}, es) is None
    assert H.place_conflict("除了甲地除醛净化器，把乙地的都关了", off, {"name": "乙地风扇"}, es) is None


def test_name_rivals_lets_through_clear_words_all_words_and_other_calls():
    for said, name in (("把戊地卧室的遮光窗帘拉上", "戊地1号卧室布帘"), ("把客厅纱帘关上", "戊地1号客厅纱帘"),
                       ("打开大净化器", "大净化器"), ("打开戊地的甲牌风扇", "戊地甲牌风扇"), ("打开己地的风扇", "己地甲牌净化风扇"),
                       ("把戊地的窗帘都关上", "戊地1号卧室布帘"),    # 说了"都"
                       ("把窗帘关上", "窗帘"),                       # HA 找不到，它自己报错
                       ("关书房灯", "戊地书房灯")):                  # 灯不归这里管
        assert _rivals(said, name) == [], said
    assert _rivals("", "戊地1号卧室布帘", recent="把戊地的窗帘关上") == []                 # 这一轮的转写还没到
    assert _rivals("把它关了", "戊地1号客厅纱帘", recent="戊地客厅的纱帘开着吗\n把它关了") == []   # 前一句说过
    area = {"area": "戊地楼下1号", "domain": ["cover"]}
    assert H.name_rivals("把戊地的窗帘关上", "把戊地的窗帘关上", "intent__HassTurnOff", area, _name_entries()) == []   # 没填 name 归区域歧义
    one = [e for e in _name_entries() if e["domain"] == "cover"][:1]
    assert H.name_rivals("关了", "关了", "intent__HassTurnOff", {"name": "卧室布帘"}, one) == []   # 这类设备只有一台


def _where_named(said, recent=None, asked=False, tool="intent__HassTurnOff", es=None):
    return H.where_named(said, said if recent is None else recent, tool, _name_entries() if es is None else es, asked=asked)


def test_where_named_asks_when_the_devices_are_in_several_places():
    for said in ("把风扇都关了", "把甲牌风扇全部打开", "所有风扇都关掉", "把风扇全关了", "把风扇都调到百分之五十", "帮我把风扇都关了吧", "风扇一起关了",
                 "都这么晚了，把风扇都关了"):
        names, question = _where_named(said)
        assert names == [] and question.startswith("没有执行") and "戊地楼下1号、己地老家" in question and "一共 2 台" in question, said
    # 设备词在前一句（"都关了吧"看前一句）；点到的不止一类
    assert "一共 2 台" in _where_named("都关了吧", "风扇有几台\n都关了吧")[1]
    assert "一共 6 台" in _where_named("把窗帘和风扇都关了")[1]
    # 更早提过地点、这句点了设备：那个地点不算（小模型照样控制几处）
    assert "一共 2 台" in _where_named("把风扇都关了", "戊地的灯开着吗\n把风扇都关了")[1]


def test_where_named_not_asked():
    for said in ("把戊地的风扇都关了", "把己地的风扇都打开", "把老家的风扇都打开", "把戊地和己地的风扇都关了",   # 说了地点
                 "把窗帘都关上", "把净化器都关了", "把大净化器都关了", "把戊地甲牌风扇都关了",                 # 只在一处、只有一台（净化风扇不带"净化器"这个词）
                 "把风扇关了", "都这么晚了，把风扇关了", "我都说了把风扇关了",                                  # 没说"都"，或者"都"不是说风扇的
                 "把书房灯都关了", "把灯全关了", "都关了吧", "全部"):                                           # 没点到名单里的设备（灯有自己的回问）
        assert _where_named(said) == ([], None), said
    # 排除、否定、提问、说状态：不是这个问题
    for said in ("除了戊地，把风扇都关了", "把风扇都关了，留一台", "别把风扇都关了", "不用把风扇都关了", "风扇都不关", "风扇都关了吗", "风扇是不是都关了",
                 "风扇都关着呢", "风扇都好久没开了", "算了风扇都关了"):
        assert _where_named(said) == ([], None), said
    assert _where_named("都关了吧", "戊地的风扇有几台\n都关了吧") == ([], None)         # 设备词那一句说了地点
    assert _where_named("戊地的都关了吧", "风扇有几台\n戊地的都关了吧") == ([], None)   # 这句说了地点
    assert _where_named("把风扇都关了", es=[]) == ([], None)                             # 名单没取到


def test_where_named_after_asking():
    here, there = "戊地108甲牌净化风扇", "己地甲牌净化风扇"

    def answer(said, first="把风扇都关了"):
        return _where_named(said, f"{first}\n{said}", asked=True)
    # 答了地点：那一处点到的设备，不管小模型填了什么
    for said, want in (("戊地的", [here]), ("己地", [there]), ("老家的吧", [there]), ("戊地和己地", [here, there]), ("戊地的风扇", [here]),
                       ("戊地的都关了吧", [here])):
        assert answer(said) == (want, None), said
    # 答了全部
    for said in ("都关", "都关了吧", "全部", "全都关掉", "一起关了", "风扇都关了"):
        assert answer(said) == ([here, there], None), said
    # 只认名单点到的：净化器答戊地，只有带这个词的那一台
    assert answer("戊地", "把净化器和风扇都关了") == ([here, "戊地108除醛净化器"], None)
    # 答的地方没有这种设备：照小模型填的，地点守卫看着
    other = [e for e in _name_entries() if e["names"][0] != there] + [{"names": ["己地窗帘"], "domain": "cover", "areas": ["己地老家"], "device_class": "curtain"}]
    assert _where_named("己地", "把甲牌风扇都关了\n己地", asked=True, es=other) == ([], None)
    # 不要了；说别的设备去了（没说"都"的是新的一句，说了"都"的按新的一句判断）
    for said in ("算了", "不用了", "都不关", "把卧室布帘拉开", "把戊地的窗帘拉开", "把除醛净化器关了", "把戊地的窗帘都拉开"):
        assert answer(said) == ([], None), said
    assert answer("窗帘都关上", "把风扇都关了") == ([], None)                        # 窗帘只在一处，也不问，小模型自己填
    assert answer("净化器和风扇都关了", "把风扇都关了")[0] == []                      # 多点了一类，是新的一句：两处都有，再问
    assert "戊地楼下1号、己地老家" in answer("净化器和风扇都关了", "把风扇都关了")[1]
    # 没答上来：再问；这一句的转写还没到也再问
    for said in ("嗯", "第一个", "好的", "风扇", "把风扇打开"):
        names, question = answer(said)
        assert names == [] and question.startswith("没有执行"), said
    assert _where_named("", "把风扇都关了", asked=True)[1].startswith("没有执行")
    # 这个工具控制不了（fan__ 只找风扇；灯的工具找不到风扇）或者没有认得准的名字：照小模型填的
    assert _where_named("戊地", "把风扇都关了\n戊地", asked=True, tool="light__HassLightSet") == ([], None)
    assert _where_named("戊地", "把风扇都关了\n戊地", asked=True, tool="fan__HassFanSetSpeed") == ([here], None)
    dup = [{"names": ["风扇"], "domain": "fan", "areas": ["甲"], "device_class": None}, {"names": ["风扇"], "domain": "fan", "areas": ["乙"], "device_class": None}]
    assert H.where_named("都关", "把风扇都关了\n都关", "intent__HassTurnOff", dup, asked=True) == ([], None)


def test_says_device_word_only_for_words_in_device_names():
    curtains = _name_entries()[:4]
    for text in ("客厅的", "把戊地的窗帘关上", "布帘"):
        assert H.says_device_word(text, curtains), text
    for text in ("第一个", "靠窗那幅", "嗯", "", "调到100"):
        assert not H.says_device_word(text, curtains), text


def test_ask_which_lists_at_most_four_names():
    es = _name_entries()
    assert H.ask_which(es[:2]) == "没有执行：用户没说是哪一台，有 2 台对得上（戊地1号卧室布帘、戊地1号卧室纱帘）。问用户要控制哪一台。"
    assert H.ask_which(es[:5], "家里").startswith("没有执行：用户没说是哪一台，家里有 5 台对得上")
    assert "戊地1号客厅纱帘等）" in H.ask_which(es[:5]) and "甲牌" not in H.ask_which(es[:5])


_CURTAINS = ["戊地1号卧室布帘", "戊地1号卧室纱帘", "戊地1号客厅布帘", "戊地1号客厅纱帘"]


def _all_named(said, recent=None, tool="intent__HassSetPosition", es=None):
    return H.all_named(said, said if recent is None else recent, tool, _name_entries() if es is None else es)


def test_all_named_lists_the_whole_kind_at_the_place():
    for said in ("把戊地的窗帘都关上", "戊地的窗帘都打开", "把戊地所有的窗帘合上", "把戊地1号的窗帘都关上", "把戊地楼下的窗帘全部打开",
                 "帮我把戊地的窗帘都关上吧", "现在把戊地的窗帘都拉开一半", "都这么晚了，把戊地的窗帘都拉上", "戊地那边的窗帘都拉上",
                 "拉上戊地所有的窗帘", "关上戊地的全部窗帘", "帮我打开戊地所有窗帘", "戊地的窗帘，都关上"):
        assert _all_named(said) == _CURTAINS, said
    # 没说地点、这类设备只在一处；说了两处，其中一处没有
    assert _all_named("把窗帘都关上") == _CURTAINS and _all_named("把家里的窗帘全部拉开") == _CURTAINS
    assert _all_named("把戊地和己地的窗帘都关上") == _CURTAINS
    # 风扇两处都有：说了地点只算那一处。fan__ 工具只找风扇
    assert _all_named("把己地的净化风扇都关了", tool="fan__HassFanSetSpeed") == ["己地甲牌净化风扇"]
    assert _all_named("把戊地和己地的甲牌风扇都关了", tool="fan__HassFanSetSpeed") == ["戊地108甲牌净化风扇", "己地甲牌净化风扇"]


def test_all_named_narrows_by_words_only_some_devices_have():
    bedroom, living = _CURTAINS[:2], _CURTAINS[2:]
    assert _all_named("把戊地卧室的窗帘都关上") == bedroom and _all_named("把卧室的窗帘都拉开") == bedroom
    assert _all_named("把戊地客厅的窗帘全部打开") == living
    assert _all_named("把戊地的纱帘都关上") == [_CURTAINS[1], _CURTAINS[3]]
    assert _all_named("把戊地的遮光窗帘都拉开") == [_CURTAINS[0], _CURTAINS[2]]
    assert _all_named("把戊地卧室的纱帘都关上") == [_CURTAINS[1]]             # 且：只有一台，调用方不展开
    # 没有共同设备的词是"或"，有共同设备的是"且"
    assert _all_named("把戊地卧室和客厅的窗帘都关上") == _CURTAINS
    assert _all_named("把戊地的布帘和纱帘都拉开") == _CURTAINS
    assert _all_named("把戊地卧室和客厅的纱帘都关上") == [_CURTAINS[1], _CURTAINS[3]]
    assert _all_named("把卧室布帘和客厅纱帘都关上") == [_CURTAINS[0], _CURTAINS[3]]    # 整个名字只算整个名字


def test_all_named_reads_the_latest_sentence_with_device_words_and_with_a_place():
    assert _all_named("都关上吧", "戊地的窗帘有几幅\n都关上吧") == _CURTAINS
    assert _all_named("都关上", "把戊地卧室的窗帘关上\n都关上") == _CURTAINS[:2]
    assert _all_named("全部", "把戊地的窗帘关上\n全部") == _CURTAINS          # 回问之后的答复：没有动作也算
    # 这句点了设备：前一句的房间不算
    assert _all_named("把戊地的窗帘都关上", "把戊地卧室的纱帘拉开\n把戊地的窗帘都关上") == _CURTAINS
    assert _all_named("窗帘也都关上", "把戊地的灯关了\n窗帘也都关上") == _CURTAINS
    # 这句没说地点：地点取前一句的（这款风扇两处都有）
    assert _all_named("甲牌风扇也都关了", "把戊地的灯关了\n甲牌风扇也都关了", tool="intent__HassTurnOff") == ["戊地108甲牌净化风扇"]
    # 这句提到的地方没有窗帘：不算前一句的设备词
    assert _all_named("把己地的窗帘都关上", "戊地的净化风扇开着吗\n把己地的窗帘都关上") == []


def test_all_named_gives_up_when_unsure():
    for said in ("除了卧室布帘，把戊地的窗帘都关上", "把戊地的窗帘都关上，纱帘不用", "把戊地的窗帘都关上，留着纱帘", "别把戊地的窗帘都关上",
                 "戊地的窗帘都关了吗", "戊地的窗帘是不是都关了", "算了戊地的窗帘都关上"):
        assert _all_named(said) == [], said                                  # 排除、否定、提问
    for said in ("把戊地主卧的窗帘都关上", "把戊地靠窗的窗帘都拉开", "把戊地楼上的窗帘都关上", "把戊地窗帘靠窗的都关上", "把戊地的帘子都关上"):
        assert _all_named(said) == [], said                                  # 名单里没有的修饰，范围可能更小
    for said in ("把戊地的都关上", "都关上吧", "把戊地的设备都关了", "把己地的窗帘都关上"):
        assert _all_named(said) == [], said                                  # 没点到那里有的设备
    for said in ("把戊地的窗帘关上", "都这么晚了，把戊地的窗帘拉上", "我都说了把戊地的窗帘拉开"):
        assert _all_named(said) == [], said                                  # 没说"都"，或者"都"不是说窗帘的
    for said in ("戊地的窗帘都坏了", "戊地的窗帘都是新换的", "戊地的窗帘都关着呢", "戊地的窗帘都好久没开了"):
        assert _all_named(said) == [], said                                  # 在说状态
    assert _all_named("都关着呢", "戊地的窗帘有几幅\n都关着呢") == []
    for said in ("把戊地的布帘都拉开，纱帘都关上", "把戊地的布帘都拉开纱帘都关上"):
        assert _all_named(said) == [], said                                  # 动作不止一种
    assert _all_named("把戊地的窗帘和净化风扇都关了") == []                  # 不止一类设备
    assert _all_named("把戊地的书房灯都关了", tool="intent__HassTurnOff") == []   # 灯不归这里管
    assert _all_named("把风扇都关了", tool="intent__HassTurnOff") == []      # 没说地点，两处都有
    assert _all_named("把戊地的窗帘都关上", tool="fan__HassFanSetSpeed") == []    # 这个工具控制不了窗帘
    assert _all_named("把戊地的窗帘都关上", es=[]) == []


def test_all_named_picks_a_name_only_this_device_has():
    def cover(*names):
        return {"names": list(names), "domain": "cover", "areas": ["戊地"], "device_class": "curtain"}
    # 第一个名字重名：用下一个；HA 拼的"设备名  实体名"不用
    es = [cover("窗帘", "东窗帘", "东窗帘  电机"), cover("窗帘", "西窗帘")]
    assert _all_named("把戊地的窗帘都关上", es=es) == ["东窗帘", "西窗帘"]
    # 找不到认得准的名字：不展开
    assert _all_named("把戊地的窗帘都关上", es=[cover("窗帘"), cover("窗帘", "西窗帘")]) == []


def test_hits_exactly_compares_what_ha_would_control():
    es, tool = _name_entries(), "intent__HassSetPosition"
    assert H.hits_exactly(tool, {"area": "戊地楼下1号", "device_class": ["curtain"]}, _CURTAINS, es)
    assert H.hits_exactly(tool, {"area": "戊地1号", "domain": ["cover"]}, _CURTAINS, es)            # 区域别名
    assert not H.hits_exactly(tool, {"area": "戊地楼下1号", "device_class": ["curtain"]}, _CURTAINS[:2], es)   # 比要的多
    assert not H.hits_exactly(tool, {"area": "戊地楼下1号", "device_class": ["shade"]}, _CURTAINS, es)         # HA 找不到
    assert not H.hits_exactly(tool, {"area": "戊地楼下1号"}, _CURTAINS, es)                         # 还会动别的
    assert not H.hits_exactly(tool, {"name": _CURTAINS[0]}, _CURTAINS, es)                          # 比要的少


def test_sounds_like_command():
    for said in ("把戊地的灯全关了", "戊地的灯全部打开", "把灯都调到百分之三十", "都关了吧", "把戊地的灯全灭了", "接着把戊地的窗帘都放下来"):
        assert H.sounds_like_command(said), said
    for said in ("戊地的灯都坏了", "戊地的灯都好久没开了", "戊地的灯都是新换的", "戊地那边的灯都亮着呢", "戊地的窗帘都关着呢", ""):
        assert not H.sounds_like_command(said), said


def test_names_listed_device_ignores_places_and_words_lights_have_too():
    for said in ("把戊地的窗帘都关上，灯打开", "把窗帘和灯都关了", "把戊地的净化风扇都关了"):
        assert H.names_listed_device(said, _name_entries()), said
    for said in ("把戊地的灯全关了", "都关了吧", "把书房灯关了", "把灯全部调到百分之三十"):
        assert not H.names_listed_device(said, _name_entries()), said
    assert not H.names_listed_device("把戊地的窗帘都关上", [])
    es = [{"names": ["卧室灯"], "domain": "light", "areas": [], "device_class": None},
          {"names": ["卧室窗帘"], "domain": "cover", "areas": [], "device_class": "curtain"}]
    assert not H.names_listed_device("把卧室的灯都关了", es) and H.names_listed_device("把窗帘都关了", es)
    # 名单里不收的（传感器）不算
    es.append({"names": ["窗台温度"], "domain": "sensor", "areas": [], "device_class": "temperature"})
    assert not H.names_listed_device("窗台温度都多少了", es)


# 甲地、老家都有客厅和卧室，厨房只有老家有
_HOUSES = ("下面是我家智能设备列表（位置，设备名，entity_id），可以通过homeassistant控制\n"
           "甲地书房,书房灯,light.a1\n甲地客厅,客厅大灯,light.a2\n甲地1号,台灯,light.a3\n甲地1号,甲牌台灯,light.a4\n甲地1号,音箱,media_player.a5\n"
           "甲地1号,灯,light.a6\n甲地1号,台灯带,light.a7\n老家卧室,卧室灯,light.b1\n老家客厅,客厅吸顶灯,light.b2\n丙地,Pro灯,light.c1\n"
           "甲地卧室,卧室顶灯,light.a8\n老家厨房,烟机灯,light.b4\n")


def _partial(said: str, es: list | None = None, prompt: str = _HOUSES) -> list[str]:
    """动作相同的句子（或者不管的）：返回设备，不回问。"""
    targets, question, differ = H.partial_names(prompt, said, entries() if es is None else es)
    assert question is None and not differ, said
    return targets


def _partial_differ(said: str, prompt: str = _HOUSES) -> list[str]:
    """每台动作不同的句子：返回设备。"""
    targets, question, differ = H.partial_names(prompt, said, entries())
    assert question is None and differ, said
    return targets


def _partial_ask(said: str, prompt: str = _HOUSES, asked: str = "") -> str:
    """定不了是哪一处的句子：不控制设备，返回回问。"""
    targets, question, differ = H.partial_names(prompt, said, entries(), asked)
    assert targets == [] and question and question.startswith("没有执行") and not differ, said
    return question


def test_partial_names_take_the_place_from_the_same_sentence():
    """"客厅灯""书房和客厅的灯"不是完整的名字，两处都有客厅：同一句里完整名字的设备在哪一处就是哪一处。"""
    for said in ("把书房灯和客厅灯关了", "把书房灯和客厅的灯关了", "把书房和客厅的灯关了", "把书房的灯和客厅的灯关了", "书房、客厅的灯都关了",
                 "帮我关一下书房和客厅的灯", "把书房灯和客厅里的灯关了", "能不能把书房和客厅的灯关了", "把书房灯关了，客厅的也关了", "把书房灯关了，客厅灯也关了",
                 "把书房灯和客厅灯关了，我要睡了", "我要睡了，关掉书房和客厅的灯", "把书房灯和客厅灯关了，把窗帘拉上", "把书房和客厅的灯调到百分之三十",
                 "可以把书房和客厅的灯关了吗", "把书房和客厅的灯关了好吗", "把书房灯和客厅灯关一下好不好", "把书房和客厅的灯亮度调到五十", "把书房和客厅的灯调暗一点儿",
                 "把书房和客厅的灯开开", "把书房客厅的灯关了"):
        assert _partial(said) == ["light.a1", "light.a2"], said
    assert _partial("把客厅灯和书房灯打开") == ["light.a2", "light.a1"]
    assert _partial("把客厅和书房的灯打开") == ["light.a2", "light.a1"]
    assert _partial("把书房客厅卧室的灯都关了") == ["light.a1", "light.a2", "light.a8"]
    # 完整的名字定地点：老家的灯带着客厅去老家；"卧室灯"这样的名字别处也可能这么叫，也跟着走
    assert _partial("把客厅灯和卧室吸顶灯关了", prompt=_HOUSES + "老家卧室,卧室吸顶灯,light.b5\n") == ["light.b2", "light.b5"]
    assert _partial("把卧室灯和客厅大灯关了") == ["light.a8", "light.a2"]
    # 片段只有一处对得上所有说法：厨房只有老家有
    assert _partial("把客厅和厨房的灯关了") == ["light.b2", "light.b4"]
    # 灯名字的片段：说了地点就在那里找；"台灯"有完整名字就是它，不算"甲牌台灯"
    assert _partial("把大灯和书房灯关了") == ["light.a2", "light.a1"]
    assert _partial("把吸顶灯和厨房灯关了") == ["light.b2", "light.b4"]
    assert _partial("把书房灯和台灯关了") == ["light.a1", "light.a3"]
    # 同一个房间里不是灯的设备不算
    assert _partial("把书房灯和客厅灯关了", prompt=_HOUSES + "甲地客厅,音箱,media_player.a9\n") == ["light.a1", "light.a2"]
    # 名字重名，同一句里分不开：问
    assert _partial_ask("把书房灯和客厅灯关了", prompt=_HOUSES + "老家书房,书房灯,light.b3\n")


def test_partial_names_with_the_place_said():
    for said in ("把老家的客厅和卧室的灯关了", "把老家的客厅灯和卧室灯关了", "把客厅和卧室的灯关了，老家的", "把老家客厅和老家卧室的灯关了"):
        assert _partial(said) == ["light.b2", "light.b1"], said
    assert _partial("把甲地的客厅灯和卧室灯关了") == ["light.a2", "light.a8"]
    # 各说各的地点
    assert _partial("把甲地的客厅灯和老家的客厅灯关了") == ["light.a2", "light.b2"]
    assert _partial("把书房灯和老家的客厅灯关了") == ["light.a1", "light.b2"]


def test_partial_names_leave_full_names_to_the_small_model():
    """都是完整的名字、查询：小模型自己做得对，这里不管。"""
    for said in ("把书房灯和客厅大灯关了", "把书房灯、客厅大灯和甲牌台灯都关了", "把书房灯和甲牌台灯关了", "把甲地的书房灯和老家的客厅吸顶灯关了", "把老家的客厅吸顶灯和烟机灯关了",
                 "把书房灯关了，再把客厅大灯打开", "书房灯和客厅大灯开着吗", "书房灯和客厅灯开着吗", "客厅和卧室的灯开着吗", "书房和客厅的灯都关了吗"):
        assert _partial(said) == [], said


def test_partial_names_place_said_for_only_one_full_name():
    """都是完整的名字，地点只说在后一台前面：地点守卫只看这句说了老家，会把书房灯拦下，所以这里定下来。"""
    assert _partial("把书房灯和老家的客厅吸顶灯关了") == ["light.a1", "light.b2"]
    assert _partial_differ("把书房灯关了，再把老家的客厅吸顶灯打开") == ["light.a1", "light.b2"]


def test_partial_names_one_word_needs_all_words():
    """只说一处：说了"都"、几处都有才回问（小模型会一次控制几处）；没说"都"小模型自己会问；只有一处的不管；同一处几盏都算。"""
    for said in ("把客厅的灯都关了", "把客厅灯全关了", "把客厅的灯全部打开"):
        assert "问用户要控制哪一处" in _partial_ask(said), said
    for said in ("把客厅灯关了", "把客厅的灯关了", "都这么晚了，把客厅灯关了", "把书房的灯都关了", "把书房灯关了", "把甲地客厅的灯都关了", "把灯都关了", "把甲地的灯都关了"):
        assert _partial(said) == [], said
    assert _partial("把吸顶灯都关了", prompt=_HOUSES + "老家卧室,卧室吸顶灯,light.b5\n") == ["light.b2", "light.b5"]
    # 两处都说了
    assert _partial("把甲地和老家的客厅灯都关了") == ["light.a2", "light.b2"]
    assert _partial("把甲地和老家的客厅灯关了") == []


def test_partial_names_ask_when_the_place_cannot_be_settled():
    # 两处都对得上，同一句里没有别的线索：不能跨房子猜
    for said in ("把客厅和卧室的灯关了", "把客厅灯和卧室灯关了", "把卧室灯和客厅灯都关了", "把客厅的灯和卧室的灯打开", "把客厅灯关了，再把卧室灯打开"):
        msg = _partial_ask(said)
        assert "甲地楼下1号" in msg and "乙地老家" in msg and "问用户要控制哪一处，还是都控制" in msg, said
    # 片段不跨房子：完整的名字可以不在一处，片段只认同一处
    for said in ("把书房灯和厨房灯关了", "把书房和厨房的灯关了", "把甲牌台灯和厨房灯关了"):
        msg = _partial_ask(said)
        assert "甲地楼下1号" in msg and "乙地老家" in msg, said
    # 说的地点没有这台：不能关到别处去
    for said in ("把老家的书房灯和客厅灯关了", "把老家的卧室灯和客厅大灯关了", "把甲地的厨房灯和书房灯关了"):
        assert _partial_ask(said), said
    # 两处都对得上时，不能各取一盏完整名字对得上的
    swap = "甲地卧室,卧室灯,light.x1\n老家卧室,卧室顶灯,light.x2\n老家客厅,客厅灯,light.x3\n甲地客厅,客厅主灯,light.x4\n"
    assert _partial_ask("把卧室灯和客厅灯关了", prompt=swap)
    # 同一处有好几盏对得上，也不替用户挑
    many = "甲地客厅,主灯,light.x1\n甲地客厅,射灯,light.x2\n甲地书房,书房灯,light.x3\n"
    assert "主灯、射灯" in _partial_ask("把书房灯和客厅灯关了", prompt=many)


def test_partial_names_answer_after_asking():
    def answer(said: str, asked: str = "把客厅和卧室的灯关了"):
        return H.partial_names(_HOUSES, said, entries(), asked)

    assert answer("甲地的") == (["light.a2", "light.a8"], None, False)
    assert answer("老家") == (["light.b2", "light.b1"], None, False)
    assert answer("甲地的客厅灯和卧室灯") == (["light.a2", "light.a8"], None, False)    # 自己说全了
    assert answer("都关") == (["light.a2", "light.b2", "light.b1", "light.a8"], None, False)
    assert answer("两边都关了吧") == (["light.a2", "light.b2", "light.b1", "light.a8"], None, False)
    # 答的地方只有其中一盏：控制那一盏；一盏都没有、没答上来（或者转写还没到）、那里还是有好几盏对得上：再问
    assert answer("甲地的", "把书房灯和厨房灯关了") == (["light.a1"], None, False)
    assert answer("甲地", "把台灯和客厅灯关了") == (["light.a3", "light.a2"], None, False)     # 那里有完整名字的就是它
    many = _HOUSES + "老家客厅,客厅筒灯,light.b6\n"
    assert H.partial_names(many, "老家", entries(), "把客厅和卧室的灯关了")[1].startswith("没有执行")
    assert H.partial_names(many, "甲地", entries(), "把客厅和卧室的灯关了")[0] == ["light.a2", "light.a8"]
    assert "light.b6" in H.partial_names(many, "都关", entries(), "把客厅和卧室的灯关了")[0]
    assert _partial_ask("老家", prompt="甲地客厅,主灯,light.x1\n甲地书房,书房灯,light.x3\n", asked="把书房灯和客厅灯关了")
    for said in ("嗯", "第一个", "", None):
        assert _partial_ask(said, asked="把客厅和卧室的灯关了"), said
    # 点了设备：照小模型的；不要了：不展开
    for said in ("就卧室灯吧", "书房灯", "厨房的", "就书房那个", "不用了", "算了"):
        assert answer(said) == ([], None, False), said
    # 问的那句动作不同、只说了一处
    assert answer("老家", "把客厅灯关了，再把卧室灯打开") == (["light.b2", "light.b1"], None, True)
    assert answer("甲地的", "把客厅的灯都关了") == (["light.a2"], None, False)
    assert answer("都关", "把客厅的灯都关了") == (["light.a2", "light.b2"], None, False)


def test_partial_names_words_that_do_not_count():
    """房间词在别的话里（"我在客厅"）、说的是别的设备（"客厅的窗帘"）、前面还有别的词（"卧室大灯""小台灯"）、有否定、没说动作：不管，也不回问。"""
    for said in ("我在客厅，把书房灯关了", "客厅太暗了，开一下书房灯", "到客厅了，把书房灯关了", "回客厅之前把书房灯关了", "把书房灯和客厅的窗帘关了",
                 "把书房和客厅的窗帘关了", "把书房灯和窗帘关了", "把书房和客厅关了", "把卧室大灯和书房灯关了", "把小台灯和书房灯关了",
                 "除了书房灯，把客厅灯关了", "把书房灯和客厅灯关了，卧室灯不用关", "不要关书房灯，把客厅灯关了", "把书房灯和客厅灯关了吗",
                 "把书房灯和客厅灯修一下", "把卧室客厅的窗帘关了", "把书房灯和卧室客厅的窗帘关了", "书房灯和客厅灯", "书房灯太亮了，把客厅灯关了",
                 "把客厅灯调到和书房灯一样亮", "书房灯开着，把客厅灯关了", "", None):
        assert _partial(said) == [], said


def test_partial_names_flag_different_actions():
    for said in ("把书房灯关了，再把客厅灯打开", "书房灯开，客厅灯关", "打开书房灯，关掉客厅灯", "开书房灯关客厅灯", "把书房灯关了，打开客厅灯", "把书房灯关了再打开客厅灯",
                 "把书房灯调亮，客厅灯调暗", "把书房灯调到三十，客厅灯调到五十", "把书房灯调大一点，客厅灯调小一点"):
        assert _partial_differ(said) == ["light.a1", "light.a2"], said
    # 同一个动作说两遍不算
    assert _partial("把书房灯调到三十，客厅灯也调到三十") == ["light.a1", "light.a2"]
    # 不同类可以：每台的动作各填各的；同一个动作不同类不管（brightness_value 不能发给音箱）
    assert _partial_differ("把客厅灯打开，音箱关了") == ["light.a2", "media_player.a5"]
    assert _partial("把客厅灯和音箱关了") == []


def test_partial_names_without_the_lists():
    # 名单没取到认不出地点，房间也认不出，"客厅灯"不算；提示词里没有设备列表就不管
    assert _partial("把书房灯和客厅灯关了", es=[]) == []
    assert _partial("把书房灯和客厅灯关了", prompt="你是小智") == []
