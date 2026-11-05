"""Live client 委托用的 Home Assistant 设备名单和地点守卫。

HA 的意图工具（MCP 里叫"平台__Hass意图"，如 intent__HassTurnOn、fan__HassFanSetSpeed）按名字找设备：先拿 name 和实体名、
别名逐个比（strip().casefold() 后全等），再拿 area 和区域名、区域别名比（helpers/intent.py 的 async_match_targets）。
区域通常只到"哪个家"一级，房间只出现在设备别名里；小模型常把"A地B房间的C"拆成 area=A、name=C，HA 必然匹配不上，
而失败结果很短、直接交给 Live 转述，没有第二次机会（11-03 空跑：智能设备列表以外的设备 52 次只对 3 次）。

所以调度指令里加一条填法规则，再带一份从 GetLiveContext 生成的名单（每台设备第一个名字 + 最短的几个别名、区域名和区域别名），
小模型照抄名字（同一批句子 51/52）。名单的副作用：HA 里有不带地名的别名时，用户说的地方没有这台设备，模型会照抄别处设备的
别名（"把乙地的X关了"关掉甲地的 X）。规则又不让把地点拆到 area，HA 自己的区域过滤也就不起作用了，所以执行前过一道地点守卫：
用户最后一句话提到的区域和这次调用会命中的设备不在一处，就不执行。再加一道区域歧义检查：没填 name、会一次控制多台，
用户又没说"都 / 全部"，也不执行、回问是哪一台（设备名没识别出来时小模型会退成只填区域）。设备歧义检查管另一半：
填了 name，但用户的话分不开这台和同类的另一台（"把甲地的窗帘关上"，小模型替用户挑了卧室那幅），同样不执行、回问。
用户说了"都"时区域歧义放行，但开关意图只填区域会动那一处所有类别的设备，所以类别歧义检查再问一次要哪一类。

用户明说要控制一类时（"把甲地的窗帘都关上"），小模型一幅一轮会用完轮数、只填区域会多控或者找不到，所以范围照名单定，每台一个调用（all_named）。
用户说了"都"却没说地点、几处都有这类设备（"把净化器都关了"），先问是哪一处，答了之后范围照名单定（where_named）。

智能设备列表里的设备（灯，走 hass_set_state）也过一道地点守卫（listed_place_conflict）：地点从列表的"位置"一栏认。

名字、区域都来自 HA，这里不写任何地名。只读：只调 GetLiveContext。
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Iterable

# 名单收哪些 domain 的设备：开关得动的那些。灯一般在 HA 插件的智能设备列表里（走 hass_set_state），默认不重复列
DEFAULT_DOMAINS = ("fan", "cover", "switch", "humidifier", "climate", "valve", "lock", "media_player", "vacuum")

LIST_HEAD = ("Home Assistant 里智能设备列表以外的可控设备（这些设备没有 entity_id，不能用 hass_set_state / hass_get_state；"
             "name 从这里整个照抄一个名字，不要自己编，也不要把两个名字拼起来）：")
AMBIGUOUS = "用户没说是哪一台、上面不止一台对得上时，不要替用户挑：name 照用户原话填。"


def norm(s: Any) -> str:
    return str(s or "").strip().casefold()


def _names(e: dict) -> list[str]:
    """跳过 HA 拼出来的"设备名  实体名"（中间两个空格）。"""
    return [n for n in e["names"] if "  " not in n] or e["names"]


def _split(v: Any) -> list[str]:
    """GetLiveContext 把多个名字用 ", " 连成一个字符串（名字本身可能带空格）。"""
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v or "").split(", ") if x.strip()]


def live_context_text(raw: Any) -> str:
    """server_mcp 执行器交回的是 str(CallToolResult)（"… content=[TextContent(type='text', text='{…}')] …"），
    从里面取出 GetLiveContext 的正文；已经是正文就原样返回。"""
    s = str(raw or "")
    m = re.search(r"text=('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")", s)
    if m:
        try:
            s = ast.literal_eval(m.group(1))
        except (ValueError, SyntaxError):
            pass
    try:
        data = json.loads(s)
    except ValueError:
        return s
    if isinstance(data, dict) and isinstance(data.get("result"), str):
        return data["result"]
    return s


def parse_live_context(raw: Any) -> list[dict]:
    """GetLiveContext → [{names, domain, areas, device_class}]；areas 第一个是区域名，后面是区域别名。认不出来返回 []。"""
    import yaml

    text = live_context_text(raw)
    i = text.find("- names:")
    if i < 0:
        return []
    try:
        data = yaml.safe_load(text[i:])
    except yaml.YAMLError:
        return []
    out = []
    for e in data if isinstance(data, list) else []:
        if not isinstance(e, dict) or not e.get("names"):
            continue
        attrs = e.get("attributes") if isinstance(e.get("attributes"), dict) else {}
        out.append({"names": _split(e.get("names")), "domain": str(e.get("domain") or ""),
                    "areas": _split(e.get("areas")), "device_class": attrs.get("device_class")})
    return out


def device_intents(tools: Iterable[dict]) -> dict[str, dict]:
    """HA 按名字 / 区域找设备的意图工具：{工具名: 工具}。名字形如"平台__Hass意图"、带 name 或 area 参数
    （HassBroadcast、HassCancelAllTimers 这类不找设备的不算）。"""
    out = {}
    for t in tools or []:
        name = str(t.get("name") or "")
        props = (t.get("parameters") or {}).get("properties") or {}
        if "__Hass" in name and ("name" in props or "area" in props):
            out[name] = t
    return out


def _required_domain(tool_name: str) -> str | None:
    """fan__HassFanSetSpeed 只找 fan，light__HassLightSet 只找 light；intent__ 开头的不限 domain。"""
    prefix = tool_name.split("__", 1)[0]
    return None if prefix == "intent" else prefix


def area_table(entries: Iterable[dict]) -> dict[str, list[str]]:
    """{区域名: [区域名, 别名…]}，按第一次出现的顺序。"""
    out: dict[str, list[str]] = {}
    for e in entries:
        if e["areas"]:
            out.setdefault(e["areas"][0], list(e["areas"]))
    return out


def dispatch_text(tools: Iterable[dict], entries: list[dict], domains: Iterable[str] = DEFAULT_DOMAINS,
                  aliases: int = 3, max_entities: int = 60) -> str:
    """拼进调度指令的一段：填法规则 + 名单 + 区域 + 有歧义时的做法。没有 HA 意图工具时返回空串；没取到名单时只给规则。"""
    tools = list(tools or [])
    intents = device_intents(tools)
    if not intents:
        return ""
    prefixes = []
    for n in sorted(intents, key=lambda n: (not n.startswith("intent__"), list(intents).index(n))):
        p = n.split("__", 1)[0] + "__"
        if p not in prefixes:
            prefixes.append(p)
    # 11-03 空跑：不写第一句时，decisions 把清单里的灯也改选 intent 工具，小模型还把清单的"地点,名字"两栏拼成一个名字去填 name
    lead = ("智能设备列表里有的设备照旧用 hass_set_state / hass_get_state。列表里没有的设备才用"
            if any(t.get("name") == "hass_set_state" for t in tools) else "控制设备时用")
    parts = [f"{lead} {'、'.join(prefixes)} 开头的 Home Assistant 工具：控制某一个设备只填 name，"
             "把用户说的地点、房间和设备名照原话连起来（用户说“A的B”就填“AB”），不要把地点或房间拆到 area，"
             "也不要填 floor、domain、device_class；控制一个地方的整类设备时才填 area 和 domain。"]
    # 查状态的 GetLiveContext 也按名字 / 别名全等过滤：11-03 外放验收"乙地的空气净化器开着吗"填了 name=空气净化器、area=乙地，没查到
    query = next((str(t["name"]) for t in tools if str(t.get("name") or "").endswith("GetLiveContext")), None)
    if query:
        parts[0] += f"查这些设备的状态用 {query}：name 也照这样填完整的设备名（名单里有就整个照抄），不要拆出 area。"
    domains = set(domains)
    rows = []
    for e in entries or []:
        if e["domain"] not in domains:
            continue
        names = _names(e)
        rows.append(" / ".join([names[0]] + sorted(names[1:], key=len)[:aliases]))
    if rows:
        parts.append(LIST_HEAD)
        parts += [f"- {r}" for r in rows[:max_entities]]
        if len(rows) > max_entities:
            parts.append(f"（还有 {len(rows) - max_entities} 台没列出）")
        areas = area_table(entries)
        if areas:
            parts.append("Home Assistant 的区域（area 只能填这些名字）：" + "；".join(
                f"{k}（也叫 {'、'.join(v[1:])}）" if v[1:] else k for k, v in areas.items()))
        parts.append(AMBIGUOUS)
    return "\n".join(parts)


# ---------- 地点守卫 ----------
def area_tokens(entries: Iterable[dict]) -> dict[str, set[str]]:
    """每个区域的地点词：区域名、区域别名，再加上只属于这个区域的别名前缀，
    比如区域"甲地楼下1号"的别名都以"甲地"开头、别的区域没有，用户只说"甲地"也认得出。"""
    raw = {k: {norm(x) for x in v if norm(x)} for k, v in area_table(entries).items()}
    pref = {k: {n[:size] for n in names for size in range(2, len(n) + 1)} for k, names in raw.items()}
    out = {}
    for k in raw:
        others = set().union(*(pref[o] | raw[o] for o in raw if o != k))
        derived = {p for p in pref[k] if p not in others}
        # 单字和纯数字的词太容易在别的话里撞上（"家""3号"），宁可认不出地点（放行）也不误拦
        out[k] = {x for x in raw[k] | derived if len(x.replace(" ", "")) >= 2 and not x.replace("-", "").isdigit()}
    return out


def mentioned_areas(text: str, tokens: dict[str, set[str]]) -> set[str]:
    t = norm(text).replace(" ", "")
    return {k for k, toks in tokens.items() if any(x.replace(" ", "") and x.replace(" ", "") in t for x in toks)}


def _as_list(v: Any) -> list:
    return v if isinstance(v, list) else [v]


def matched(tool_name: str, args: dict, entries: Iterable[dict]) -> list[dict]:
    """粗算这次调用在 HA 上会命中哪些实体：domain（或这个意图限定的 domain）、名字 / 别名全等、device_class、区域名 / 别名全等。"""
    a = {k: v for k, v in (args or {}).items() if v is not None and not (isinstance(v, str) and not v.strip())}
    req = _required_domain(tool_name)
    doms = set(map(str, _as_list(a["domain"]))) if "domain" in a else ({req} if req else None)
    cands = [e for e in entries if doms is None or e["domain"] in doms]
    name = a.get("name")
    if isinstance(name, str) and norm(name) != "all":
        cands = [e for e in cands if any(norm(x) == norm(name) for x in e["names"])]
    dcs = set(map(str, _as_list(a["device_class"]))) if a.get("device_class") else None
    if dcs:
        cands = [e for e in cands if e.get("device_class") in dcs]
    area = a.get("area")
    if isinstance(area, str):
        cands = [e for e in cands if any(norm(x) == norm(area) for x in e["areas"])]
    return cands


def place_conflict(utterance: str, tool_name: str, args: dict, entries: list[dict]) -> str | None:
    """用户这句话提到了某个区域、这次调用却会命中别的区域的设备：返回交给 Live 的说明（不执行）；没问题返回 None。
    没提地点、HA 本来就匹配不上、命中的设备没分区域，都放行。"""
    said = mentioned_areas(utterance, area_tokens(entries))
    if not said:
        return None
    off = [e for e in matched(tool_name, args, entries) if e["areas"] and e["areas"][0] not in said]
    if not off:
        return None
    where = "、".join(sorted({e["areas"][0] for e in off}))
    return (f"没有执行：用户说的是{'、'.join(sorted(said))}，名字对得上的设备在{where}，不是同一个地方。"
            "告诉用户那里没有这台设备，或者问清是哪一台。")


# 用户明说要控制一类设备的词。"全"只认"全都 / 全部 / 全屋"和后面紧跟动作的（全关、全打开）：
# 11-03 外放验收里"除甲醛"被转写成"出价全"，光认"全"会把这种转写错的设备名当成"全部"
_ALL_WORDS = re.compile(r"都|全部|所有|全都|全屋|一起|统统|全(?=[关开打拉调停合灭熄])")
_NOT_ACTIONABLE = {"sensor", "binary_sensor"}


def says_all(text: str) -> bool:
    """用户有没有明说要控制一类设备（都 / 全部 / 所有 / 全关……）。"""
    return bool(_ALL_WORDS.search(text or ""))


def ask_which(hits: list[dict], where: str = "") -> str:
    """交给 Live 的说明：没有执行，问用户是哪一台。"""
    names = "、".join(e["names"][0] for e in hits[:4]) + ("等" if len(hits) > 4 else "")
    return f"没有执行：用户没说是哪一台，{where}有 {len(hits)} 台对得上（{names}）。问用户要控制哪一台。"


def area_ambiguity(utterance: str, tool_name: str, args: dict, entries: list[dict]) -> str | None:
    """没填 name（只填区域 / 类别）的控制调用：用户没说"都 / 全部 / 所有 / 全…"、会命中的可控设备又不止一台时，
    返回交给 Live 的说明（不执行，回问是哪一台）；只有一台、用户明说了要全部、或者填了 name，都放行。
    设备名没识别出来时，小模型会退成只填 area + domain，一次控制那个地方的同类设备（"乙地的风扇风速调到最大"
    把同处的净化器也调了）；"把书房的灯关了"这种区域里只有一盏的照常执行。"""
    a = {k: v for k, v in (args or {}).items() if v is not None and not (isinstance(v, str) and not v.strip())}
    if isinstance(a.get("name"), str) and norm(a["name"]) != "all":
        return None
    if says_all(utterance):
        return None
    hits = [e for e in matched(tool_name, a, entries) if e["domain"] not in _NOT_ACTIONABLE]
    if len(hits) <= 1:
        return None
    return ask_which(hits, f"{a['area']}" if isinstance(a.get("area"), str) else "家里")


# HassTurnOn / HassTurnOff 不限 domain：对区域里每个实体都动手（灯、风扇开关，窗帘开合，门锁上锁 / 开锁），没有开关服务的单台失败、不影响别的
# （HA 的 OnOffIntentHandler）。HassSetPosition / HassStopMoving 碰到窗帘、阀门以外的实体整个调用就失败了，不会多控，不归这里管
_ON_OFF_INTENTS = {"HassTurnOn", "HassTurnOff"}
_NO_ON_OFF = _NOT_ACTIONABLE | {"select", "number", "text", "date", "time", "datetime", "event", "update", "weather",
                                "image", "device_tracker", "person"}
_KIND_NAMES = {"light": "灯", "fan": "风扇或净化器", "cover": "窗帘", "switch": "开关", "humidifier": "加湿器", "climate": "空调",
               "media_player": "音箱或电视", "vacuum": "扫地机", "lock": "门锁", "valve": "阀门"}


def would_switch(tool_name: str, args: dict, entries: Iterable[dict]) -> list[dict]:
    """这次调用会命中的实体里，HA 真会动手的那些（传感器之类去掉）。"""
    return [e for e in matched(tool_name, args, entries) if e["domain"] not in _NO_ON_OFF]


def mixed_kinds(tool_name: str, args: dict, entries: list[dict]) -> list[dict]:
    """开关意图没填 domain、device_class，又没有 name 把范围收窄：HA 会动那一处所有能开关的设备。不止一类时返回这些设备，否则返回 []。
    "把甲地的净化器都关了"小模型偶尔只填 area（"都"让 area_ambiguity 放行），净化器以外的灯、窗帘也一起动，结果里还只说净化器关了。"""
    a = {k: v for k, v in (args or {}).items() if v is not None and not (isinstance(v, str) and not v.strip())}
    if tool_name.rpartition("__")[2] not in _ON_OFF_INTENTS or a.get("domain") or a.get("device_class"):
        return []
    hits = would_switch(tool_name, a, entries)
    return hits if len({e["domain"] for e in hits}) > 1 else []


def ask_kind(hits: list[dict], args: dict) -> str:
    """交给 Live 的说明：没有执行，问用户要控制哪一类。写清这次调用会动到什么，用户已经说了"灯"时 Live 也能接得上。"""
    area = (args or {}).get("area")
    counts: dict[str, int] = {}
    for e in hits:
        counts[e["domain"]] = counts.get(e["domain"], 0) + 1
    kinds = "、".join(f"{_KIND_NAMES.get(d, d)} {n} 个" for d, n in counts.items())
    return (f"没有执行：这样会动到{area if isinstance(area, str) else '家里'}所有能开关的设备（{kinds}）。"
            "问用户要控制哪一类，还是全部。")


# 用户答"哪一类"时点了类别的词："窗帘都关了"不是全部
_KIND_WORDS = re.compile(r"灯|帘|风扇|净化|开关|加湿|空调|音箱|电视|扫地|门锁|阀门")


def said_all_kinds(utterance: str) -> bool:
    """用户最后这句话有没有说全部（回问"哪一类"之后答"都关"）。"""
    last = (utterance or "").rsplit("\n", 1)[-1]
    return says_all(last) and not _KIND_WORDS.search(last)


def _name_parts(e: dict) -> set[str]:
    """设备各个名字里长度 ≥ 2 的连续片段（转小写、去空格）。纯数字不要：别名里的门牌号（"108"）会撞上"调到100"。"""
    parts: set[str] = set()
    for n in _names(e):
        t = norm(n).replace(" ", "")
        parts |= {t[i:j] for i in range(len(t)) for j in range(i + 2, len(t) + 1)}
    return {x for x in parts if not x.isdigit()}


def says_device_word(text: str, devices: Iterable[dict]) -> bool:
    """这句话里有没有这些设备名字的片段。"""
    t = norm(text).replace(" ", "")
    return any(x in t for e in devices for x in _name_parts(e))


def name_rivals(utterance: str, recent: str, tool_name: str, args: dict, entries: list[dict],
                domains: Iterable[str] = DEFAULT_DOMAINS) -> list[dict]:
    """填了 name、只命中一台的控制调用：用户说没说清是这一台。对同 domain 的每一台别的设备，用户最近的话（recent）里
    要有一个词只在这一台的名字里、不在那一台的名字里，才算分开了。地名不用单独处理：同一处的设备都带它，分不开；
    别处的设备不带，用户说了就分开了。全都分开返回 []；否则返回这一台和分不开的那几台（按名单顺序）。

    只判名单里的 domain（灯走 hass_set_state，不归这里管）；用户说了"都 / 全部"、这一轮一句都还没听到（utterance 为空）、
    HA 本来就找不到或有重名，都放行。11-03 空跑："把甲地的窗帘关上"小模型每次都替用户挑一幅，规则里写了不要挑也没用。"""
    a = {k: v for k, v in (args or {}).items() if v is not None and not (isinstance(v, str) and not v.strip())}
    if not isinstance(a.get("name"), str) or norm(a["name"]) == "all":
        return []
    if not (utterance or "").strip() or says_all(utterance):
        return []
    hits = matched(tool_name, a, entries)
    if len(hits) != 1 or hits[0]["domain"] not in set(domains):
        return []
    e, text = hits[0], norm(recent).replace(" ", "")
    mine = _name_parts(e)
    group = [s for s in entries if s is e or (s["domain"] == e["domain"] and not any(x in text for x in mine - _name_parts(s)))]
    return group if len(group) > 1 else []


_EXCEPT_WORDS = re.compile(r"除|别|不|留|算了|吗")


def _device_rows(prompt: str) -> list[list[str]]:
    """提示词的智能设备列表：每行 [位置, 设备名, entity_id]。"""
    return [p for p in ([x.strip() for x in line.split(",")] for line in (prompt or "").splitlines())
            if len(p) == 3 and re.fullmatch(r"\w+\.\w+", p[2])]


def listed_place_conflict(prompt: str, said: str, entity_id: str, entries: list[dict]) -> str | None:
    """hass_set_state 的地点守卫（place_conflict 管 HA 意图，这里管智能设备列表里的设备）：用户这句话（said）提到了地点、这台设备的位置却在
    别处，返回交给小模型 / Live 的说明（不执行）；没问题返回 None。11-05 模拟："把甲地的灯都关了""把乙地的灯也都关了"，第一轮展开乙地的灯都对，
    第二轮小模型又把甲地的灯调了一遍（答复里说"之前说关了，其实没真的去关"：调度只看得到这次委托的工具记录，看不到上一次的）。

    said 只给最后一句：上一句常落在委托标记之后（交回之后 Live 才开口），把它也算上，上一句的地点就放行了。地点只看带动作的小句：
    "乙地今天多少度，顺便把书房灯关了"说乙地的是天气；一个带动作的小句都没有（"那乙地的呢"）才看整句。
    拿不准都放行：要控制的那句没提地点（"都关了吧"看前一句）；有排除、否定、提问的词（"除了甲地，把灯都关了"，提到的地点正好不是要控制的）；
    设备不在列表里；位置里认不出地点。"""
    if _EXCEPT_WORDS.search(said or ""):
        return None
    tokens = area_tokens(entries)
    places = mentioned_areas("，".join(c for c in _CLAUSE.split(said or "") if _COMMAND_WORDS.search(c)) or said, tokens)
    if not places:
        return None
    row = next((p for p in _device_rows(prompt) if p[2] == entity_id), None)
    at = mentioned_areas(row[0], tokens) if row else set()
    if not at or at & places:
        return None
    where = "、".join(sorted(places))
    return (f"没有执行，这不是出错：用户这句说的是{where}，这台设备在{row[0]}，这句没有要控制它，不用重做，也不用提它。"
            f"只说{where}的结果；{where}还没控制的话接着控制，或者问清是哪一台。")


def all_at_place(prompt: str, said: str, recent: str, entity_id: str, entries: list[dict],
                 asked: bool = False) -> tuple[list[str], str | None]:
    """"把甲地的灯全关了"要控制哪些设备，返回 (设备, 交给 Live 的回问)。设备是提示词的智能设备列表（每行"位置,设备名,entity_id"）里，
    和 entity_id 同类、位置属于用户说的地点的全部设备；地点取手边的话（recent，一行一句）里最近一句提到地点的（"都关了吧"看前一句）。

    这里定了就不再问小模型，所以拿不准都返回 ([], None)，照旧只控制小模型填的那一个：用户这句话（said）点了具体设备（"都这么晚了，
    把书房灯关了"，"都"不是全部的意思）或者有排除、否定、提问的词（"除了甲地""乙地的不用""都不关""都关了吗"）。

    哪句都没有地点（"把灯全关了"）、这类设备又不止在一处：不替用户把几处房子的都控制了，先回问。asked 表示上一次委托问过：这次答了地点
    就控制那一处，答"都关"才控制全部，没答上来（"嗯""第一个"，或者转写还没到）再问一遍。11-04 模拟：问完用户只说"嗯"，小模型一轮就把
    三处的 10 盏灯都调了，所以这里连小模型一轮出好几个调用的也拦。"""
    rows = _device_rows(prompt)
    if _EXCEPT_WORDS.search(said or "") or says_device_word(said, [{"names": [p[1]]} for p in rows]):
        return [], None
    same = [p for p in rows if p[2].split(".")[0] == entity_id.split(".")[0]]
    tokens = area_tokens(entries)

    def located(places: Iterable[str]) -> list[str]:
        words = {w.replace(" ", "") for a in places for w in tokens[a]}
        return [p[2] for p in same if any(w in norm(p[0]).replace(" ", "") for w in words)]

    places = next(filter(None, (mentioned_areas(t, tokens) for t in reversed((recent or "").split("\n")))), set())
    if places:
        return located(places), None
    where = [a for a in tokens if located([a])]
    if len(where) <= 1:
        return located(where), None
    if asked and says_all(said):
        return [p[2] for p in same], None
    return [], _ask_where(where, len(same))


def tidy_args(tool: dict, args: dict, entries: list[dict]) -> dict:
    """domain 里填了 HA 没有的 domain（把窗帘填成 curtain 之类）：是这个工具 device_class 枚举里的值就挪到 device_class，
    否则丢掉；HA 有的 domain 照留。11-03 空跑："把X的窗帘都关上"常填 domain: ["curtain"]，HA 按 domain 一个都找不到。"""
    dom = (args or {}).get("domain")
    known = {e["domain"] for e in entries or []}
    if not dom or not known:
        return args
    dl = [str(d) for d in _as_list(dom)]
    bad = [d for d in dl if d not in known]
    if not bad:
        return args
    enum = ((((tool.get("parameters") or {}).get("properties") or {}).get("device_class") or {}).get("items") or {}).get("enum") or []
    out = dict(args)
    keep = [d for d in dl if d in known]
    if keep:
        out["domain"] = keep
    else:
        out.pop("domain", None)
    moved = [d for d in bad if d in enum]
    if moved:
        dc = [str(x) for x in _as_list(out.get("device_class") or [])]
        out["device_class"] = dc + [d for d in moved if d not in dc]
    return out


# HA 意图工具里指定设备的参数，其余的是动作（position、percentage 这些）
TARGET_ARGS = {"name", "area", "floor", "domain", "device_class"}
# 说一整类设备时，设备词前面能出现的话；别的修饰（"主卧的窗帘""靠窗的窗帘"，名单里没有这个词）范围可能更小，服务端看不出来。
# "把"前面的不算修饰（"现在把……"），没有"把"时句首可以是动作（"拉上甲地所有的窗帘"）
_BA = re.compile(r"[把将]")
_LEAD_VERB = re.compile(r"^(?:帮我|给我|帮忙|请|麻烦你?|我要|我想|你)*(?:关上|关掉|关闭|拉上|拉开|打开|合上|关|开)")
_FILLER = re.compile(r"[的里家屋和跟与及也、]|那边|这边|还有")
_CLAUSE = re.compile(r"[，,。；;！!？?]")


_COMMAND_WORDS = re.compile(r"[关开打拉调灭熄合停收放升降]")
_STATIVE_WORDS = re.compile(r"没|[关开拉合打亮]着")


def sounds_like_command(said: str) -> bool:
    """这句话像不像让设备动：有动作的字，也不是在说状态（"甲地的灯都坏了""好久没开了""都关着呢"）。"""
    return bool(_COMMAND_WORDS.search(said or "")) and not _STATIVE_WORDS.search(said or "")


def _without_places(text: str, tokens: dict[str, set[str]]) -> str:
    """去掉地点词（长的先去），转小写、去空格。"""
    t = norm(text).replace(" ", "")
    for w in sorted({w.replace(" ", "") for ws in tokens.values() for w in ws}, key=len, reverse=True):
        t = t.replace(w, "")
    return t


def _named_by(text: str, group: list[dict]) -> tuple[set[str], list[dict]]:
    """text 里点到了 group 名字里的哪些词（只算最长的），这些词圈定哪几台。几个词一般是"且"（"卧室的纱帘"），
    两个词没有共同的设备时是"或"（"卧室和客厅""布帘和纱帘"）。"""
    parts = [_name_parts(e) for e in group]
    found = {w for p in parts for w in p if w in text}
    words = {w for w in found if not any(w != o and w in o for o in found)}
    has = {w: {i for i, p in enumerate(parts) if w in p} for w in words}
    return words, [e for i, e in enumerate(group)
                   if all(i in has[w] or any(i in has[o] and not has[o] & has[w] for o in words) for w in words)]


def _only_these_words(text: str, words: set[str]) -> bool:
    """设备词（words）前面除了"都 / 全部"和地点（已去掉）没有别的修饰，前面的分句也没说别的设备（"布帘都拉开，纱帘都关上"动作不止一种）。"""
    pat = re.compile("|".join(map(re.escape, sorted(words, key=len, reverse=True))))
    end = max(m.start() for m in [*pat.finditer(text), *_ALL_WORDS.finditer(text)])
    *before, head = _CLAUSE.split(text[:end])
    if pat.search(head) and any(pat.search(c) for c in before):
        return False
    return not _FILLER.sub("", _LEAD_VERB.sub("", _ALL_WORDS.sub("", pat.sub("", _BA.split(head)[-1]))))


def all_named(said: str, recent: str, tool_name: str, entries: list[dict], domains: Iterable[str] = DEFAULT_DOMAINS) -> list[str]:
    """"把甲地的窗帘都关上"要控制哪些设备，返回每台设备让 HA 认得准的名字。范围只看用户的话和名单：地点取手边的话（recent，一行一句）里
    最近一句提到地点的；设备词取最近一句点了设备的（"都关上吧"看前一句），只有一部分设备带的词会缩小范围（"卧室的窗帘""纱帘"）。

    这里定了就不再问小模型，所以拿不准都返回 []，照旧执行小模型填的：用户这句话（said）没说"都 / 全部"（"把"前面的不算："都这么晚了，
    把甲地的窗帘拉上"）；有排除、否定、提问的词；在说状态（"都坏了""都关着呢"；回问之后只答"全部"的不算，它没点设备）；没点到名单里的词，
    或者还有名单里没有的修饰；点到的不止一类设备（一个工具一个动作）；没说地点、点到的设备又不止在一处；有一台没有 HA 认得准的名字。"""
    if not says_all(_BA.split(said or "")[-1]) or _EXCEPT_WORDS.search(said or "") or _STATIVE_WORDS.search(said or ""):
        return []
    devices = [e for e in entries if e["domain"] in set(domains)]
    tokens = area_tokens(entries)
    if not _COMMAND_WORDS.search(said) and says_device_word(_without_places(said, tokens), devices):
        return []
    lines = (recent or "").split("\n")
    places = next(filter(None, (mentioned_areas(t, tokens) for t in reversed(lines))), set())
    text = next((t for t in (_without_places(x, tokens) for x in reversed(lines)) if says_device_word(t, devices)), "")
    words, hits = _named_by(text, [e for e in devices if not places or (e["areas"] and e["areas"][0] in places)])
    if not words or len({e["domain"] for e in hits}) != 1 or not _only_these_words(text, words):
        return []
    if not places and len({e["areas"][0] if e["areas"] else "" for e in hits}) > 1:
        return []
    names = [next((n for n in _names(e) if matched(tool_name, {"name": n}, entries) == [e]), None) for e in hits]
    return [] if None in names else names


def hits_exactly(tool_name: str, args: dict, names: list[str], entries: list[dict]) -> bool:
    """这个调用在 HA 上命中的正好是 names 这几台（小模型只填区域也填对了）。"""
    want = [e for n in names for e in matched(tool_name, {"name": n}, entries)]
    hits = matched(tool_name, args, entries)
    return len(hits) == len(want) and all(e in want for e in hits)


def _ask_where(where: list[str], count: int) -> str:
    """交给 Live 的说明：没有执行，问用户要控制哪一处。"""
    return f"没有执行：用户没说是哪一处，{'、'.join(where)}都有这类设备，一共 {count} 台。问用户要控制哪一处，还是全部。"


def where_named(said: str, recent: str, tool_name: str, entries: list[dict], domains: Iterable[str] = DEFAULT_DOMAINS,
                asked: bool = False) -> tuple[list[str], str | None]:
    """"把净化器都关了"走 HA 意图：用户说了"都"（"把"前面的不算："都这么晚了，把净化器关了"），点到的设备不止在一处，话里又没说地点，
    就不替用户把几处房子的都控制了，和灯一样先问，返回 ([], 交给 Live 的回问)。11-05 模拟：Haiku 不问，27 次里 23 次控制了几处房子，
    "把风扇都打开"填 domain: fan 连净化器也开了。设备词取手边的话（recent，一行一句）里最近一句点了设备的，地点只看这一句和它后面的话：
    更早提过地点不算（"乙地今天多少度""把风扇都打开"，小模型照样控制三处，3/3）。

    asked 表示上一次委托问过。这句答了地点返回那一处点到的设备，答"都关"返回全部（每台一个 HA 认得准的名字）：范围照名单定，小模型只填工具和动作。
    只答地点时小模型常填"乙地净化器"这种 HA 找不到的名字（模拟 24 次里 8 次）。只算名字里有这个词的：答"甲地"时名字里没有"净化器"的净化风扇不算。
    没答上来（"嗯""第一个"，或者转写还没到）再问一遍；答复里点了别的设备（"把甲地的窗帘拉开"）是新的一句，不算在答。

    其余返回 ([], None)，照旧执行小模型填的：排除、否定、提问、说状态（"除了甲地""别都关""都关了吗""都关着呢"，Haiku 自己做得对，模拟排除 45 次 44 次全对、否定 18 次都没动）；
    没点到名单里的设备；这几处只有一处有；有一台没有 HA 认得准的名字。"""
    said = said or ""
    if not (asked or says_all(_BA.split(said)[-1])) or _EXCEPT_WORDS.search(said) or _STATIVE_WORDS.search(said):
        return [], None
    devices = [e for e in entries if e["domain"] in set(domains)]
    tokens = area_tokens(entries)
    lines = (recent or "").split("\n")
    texts = [_without_places(x, tokens) for x in lines]
    at = next((i for i in reversed(range(len(lines))) if says_device_word(texts[i], devices)), None)
    if at is None:
        return [], None
    places = next(filter(None, (mentioned_areas(x, tokens) for x in reversed(lines[at:]))), set())
    group = [e for e in devices if not places or (e["areas"] and e["areas"][0] in places)]
    words, hits = _named_by(texts[at], group)
    if not words:
        return [], None
    if asked and lines[at] == said:
        # 答复里点了设备：和上一句同一批才算在答（"甲地的净化器"）
        earlier = next((t for t in reversed(texts[:at]) if says_device_word(t, devices)), "")
        asked = words <= _named_by(earlier, group)[0]
        if not (asked or says_all(_BA.split(said)[-1])):
            return [], None
    answered = asked and (bool(places) or says_all(said))
    where = list(dict.fromkeys(e["areas"][0] for e in hits if e["areas"]))
    if not (answered or places) and len(where) > 1:
        return [], _ask_where(where, len(hits))
    if not answered:
        return [], None
    names = [next((n for n in _names(e) if matched(tool_name, {"name": n}, entries) == [e]), None) for e in hits]
    return ([] if None in names else names), None


def names_listed_device(said: str, entries: list[dict], domains: Iterable[str] = DEFAULT_DOMAINS) -> bool:
    """用户这句话点了名单里（智能设备列表以外）的设备：地点词和灯的名字里也有的词不算（"窗帘"算，"卧室"不算）。"""
    text = _without_places(said, area_tokens(entries))
    lights = set().union(*(_name_parts(e) for e in entries if e["domain"] == "light"))
    return any(w in text for e in entries if e["domain"] in set(domains) for w in _name_parts(e) - lights)


# ---------- 不完整的说法（"把书房和客厅的灯关了"） ----------
# 设备名之间和前后不算动作的字：连接词、语气词，以及"关掉 / 关闭 / 关上""打开"和"关""开"的差别。问句在这之前已经排除了，走到这里的"吗"是客气的问法
_NOT_ACTION = re.compile(r"[\s，,、。；;！!：:]|以及|还有|一起|全部|全都|统统|所有|然后|接着|麻烦|一下|那个|这个"
                         r"|和|跟|与|及|同|并|把|将|给|帮|请|你|我|也|都|全|再|先|还|了|吧|啊|呀|哦|嘛|呗|哈|呢|啦|咯|喽|儿|掉|闭|上|打"
                         r"|好不好|行不行|可以吗|好吗|行吗|吗")
# 客气的问法：这里的"不""吗"不是否定，也不是在问状态
_POLITE_ASK = re.compile(r"能不能|可不可以|行不行|好不好|好吗|行吗|(?:能|可以).*吗")
# 去掉 _NOT_ACTION 之后，一段话整个是这些才算动作。"太亮""开着""一样亮"不算：这时设备只是被提到，不一定要控制
_ACTION = re.compile(r"开+|关|灭|熄灭|点亮|[亮暗](?:一?点|一些|些)?"
                     r"|(?:亮度|色温|颜色)?(?:(?:调|开|换|变|设置?)[到成为至].+|调[亮暗高低大小](?:一?点|一些|些)?)")
# 单说房间（没带"灯"）后面接这些才算点到：别的话（"客厅太暗了""客厅的窗帘"）里的房间不算
_AFTER_ROOM = re.compile(r"$|[和跟与及同、，,；;。！!也都全一再还并以]|关|开|打|调|灭|熄|亮|暗")


def _squash(text: Any) -> str:
    """比设备名用：转小写、去空格和"的"（"书房的灯"也算"书房灯"）。"""
    return norm(text).replace(" ", "").replace("的", "")


def _spots(rows: list[list[str]], tokens: dict[str, set[str]]) -> list[dict]:
    """设备列表的每一行拆成 {id, name, house, room}：house 是位置里带的地点词属于哪个区域，room 是位置去掉地点词之后剩下的。
    认不出 house 的（名单没取到，或者位置只写了房间）都算同一处。"""
    out = []
    for pos, name, entity in rows:
        where = _squash(pos)
        house = next((a for a, ws in tokens.items() if any(_squash(w) in where for w in ws)), "")
        room = where
        for w in sorted(map(_squash, tokens.get(house, ())), key=len, reverse=True):
            room = room.replace(w, "") if w else room
        out.append({"id": entity, "name": _squash(name), "house": house, "room": room})
    return out


def _lamp_words(spots: list[dict]) -> dict[str, dict]:
    """用户点一台设备可能的说法 → {rows: 对得上的设备, exact: 名字就是这个的设备, room: 是房间的说法}：完整的名字；房间 + 灯（"客厅灯"，这时"卧室灯"这样的
    完整名字也可能是别处的卧室）；灯的名字的片段（"台灯""筒灯"）；单说房间。"""
    words: dict[str, dict] = {}

    def add(word: str, spot: dict, exact: bool = False, room: bool = False) -> None:
        w = words.setdefault(word, {"rows": [], "exact": [], "room": False})
        if spot not in w["rows"]:
            w["rows"].append(spot)
        if exact:
            w["exact"].append(spot)
        w["room"] = w["room"] or room

    for s in spots:
        name, lamp = s["name"], "灯" in s["name"] or s["id"].startswith("light.")
        if len(name) >= 2:
            add(name, s, True)
        if lamp and len(s["room"]) >= 2:
            add(s["room"] + "灯", s, room=True)
            add(s["room"], s, room=True)
        for i, j in ((i, j) for i in range(len(name)) for j in range(i + 2, len(name) + 1)):
            if "灯" in name[i:j]:
                add(name[i:j], s)
    return words


def _mentions(text: str, words: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """按说的先后找这句话点到的设备，返回 (每一处 {word, rows, exact}, 去掉这些之后剩下的几段话)。长的说法优先（"客厅大灯"不算"客厅"）。"""
    if not words:
        return [], [text]
    found, pieces, at = [], [], 0
    pattern = re.compile("|".join(map(re.escape, sorted(words, key=len, reverse=True))))
    for m in pattern.finditer(text):
        w = words[m.group()]
        # 单说房间：后面要接连接词、动作，或者紧跟下一个房间（"书房客厅的灯"）；"卧室大灯"的"卧室"不算
        after = pattern.match(text, m.end())
        if "灯" not in m.group() and not w["exact"] and not (_AFTER_ROOM.match(text, m.end()) or (after and words[after.group()]["room"])):
            continue
        found.append(dict(w, word=m.group()))
        pieces.append(text[at:m.start()])
        at = m.end()
    return found, pieces + [text[at:]]


def _ask_place(found: list[dict]) -> str:
    """定不了是哪一处：交给 Live 的回问。"""
    said = "、".join(dict.fromkeys(m["word"] for m in found))
    spots = list({s["id"]: s for m in found for s in m["rows"]}.values())
    houses = "、".join(dict.fromkeys(s["house"] for s in spots if s["house"]))
    return (f"没有执行：用户说的{said}定不了是哪一处，对得上的设备" + (f"在{houses}" if houses else "") +
            f"（{'、'.join(s['name'] for s in spots[:6])}）。问用户要控制哪一处，还是都控制。")


def _settle(found: list[dict], bound: list[set[str]]) -> list[str] | None:
    """每一处点到的设备定到一台。bound 是这一处前面说的地点；没说地点的，看同一句里完整名字的设备或说了的地点在哪一处；
    都没有的，看有没有一处能对上所有说法（"书房和客厅的灯"，书房只有一处）。定不了返回 None，不跨房子猜。
    完整的名字各算各的，可以不在一处（"书房灯和乙地的门灯"），片段只认这同一处。"""
    rows = [[s for s in m["rows"] if not b or s["house"] in b] for m, b in zip(found, bound)]
    if not all(rows):
        return None
    sure = [m["rows"] == m["exact"] for m in found]
    anchors = set().union(*bound, *({r[0]["house"]} for r, ok in zip(rows, sure) if ok))
    if not anchors:
        anchors = set.intersection(*({s["house"] for s in r} for r in rows))
        if len(anchors) != 1:
            return None
    out = []
    for m, r, ok in zip(found, rows, sure):
        r = r if ok else [s for s in r if s["house"] in anchors]
        if len(r) > 1:
            r = [s for s in r if s in m["exact"]] or r
        if len(r) != 1:
            return None
        out.append(r[0]["id"])
    return out


def _action_words(pieces: list[str]) -> list[str] | None:
    """设备名前后这几段话里的动作；有不是动作的话返回 None。第一个设备之前只看最近的一小句，最后一个设备之后只看紧跟着的一小句：
    "我要睡了，关掉书房灯和客厅灯""把书房灯和客厅灯关了，把窗帘也拉上"多出来的那句不是在说这几台。"""
    out = []
    for n, piece in enumerate(pieces):
        parts = re.split(r"[，,。；;！!：:]", piece)
        if n == 0:
            parts = parts[-1:]
        elif n == len(pieces) - 1:
            parts = next((parts[i:i + 1] for i, x in enumerate(parts) if _NOT_ACTION.sub("", x)), [])
        for part in parts:
            for x in re.split(r"了|再|然后|接着", part):
                w = _NOT_ACTION.sub("", x)
                if w and not _ACTION.fullmatch(w):
                    return None
                if w:
                    out.append(w)
    return out


def partial_names(prompt: str, said: str, entries: list[dict], asked: str = "") -> tuple[list[str], str | None, bool]:
    """"把书房灯和客厅灯关了""把书房和客厅的灯关了"：说的不是设备列表里的完整名字，返回 (要控制的设备, 交给 Live 的回问, 每台动作不同)。
    都是完整名字的句子小模型自己做得对（11-05 模拟点名两盏 30/30，地点只说在其中一台前面的除外），查询也对（12/12，只读，两处都查也不要紧），这里不管。不完整的说法它会出错，同一批模拟：
    先做有把握的那盏，短结果直接交回，另一盏没做还说做完了（"把书房灯和客厅灯关了"0/3）；两处都有客厅时一起控制（"把客厅和卧室的灯关了"3/3 一次四盏）。

    认的说法：完整的名字、房间 + 灯（"客厅灯"）、单说房间（"书房和客厅的灯"）、灯名字的片段（"台灯""筒灯"）。至少两处；只说一处时要说了"都"（"把客厅的灯都关了"），
    没说"都"小模型自己会问。定是哪一处见 _settle：说了的地点、完整名字的设备、一处能对上所有说法。定不了（两处都对得上、片段不在一处、说了地点那里没有）就不跨房子猜，
    返回回问。上一次委托问过（asked 是问的那句话）时，这次答了地点就是那一处的，答"都关"就是这几个说法对得上的全部，点了设备照小模型的，没答上来再问。

    拿不准都返回 ([], None, False)，照旧执行小模型填的：有排除、否定、提问的词（"除了书房灯""卧室灯不用关""……关了吗"）；设备名前后还有不是动作的话
    （"卧室大灯""客厅的窗帘""我在客厅""客厅太暗了""把客厅灯调到和书房灯一样亮"）；这几台不同类又是同一个动作。名单没取到时认不出地点，房间也认不出。"""
    text = said or ""
    rows = _device_rows(prompt)
    tokens = area_tokens(entries)
    words = _lamp_words(_spots(rows, tokens))
    found, pieces = _mentions(_squash(text).replace("里灯", "灯"), words)
    answering = bool(asked) and len(found) < 2
    if _EXCEPT_WORDS.search(_POLITE_ASK.sub("", text)):
        return [], None, False
    if answering:
        if found or says_device_word(text, [{"names": [p[1]]} for p in rows]):
            return [], None, False
        found, pieces = _mentions(_squash(asked).replace("里灯", "灯"), words)
    if not found or all("灯" not in m["word"] and not m["exact"] for m in found):
        return [], None, False
    if len(found) == 1 and not says_all(_BA.split(asked if answering else text)[-1]):
        return [], None, False
    bound, last = [], set()
    for m, piece in zip(found, pieces):
        last = mentioned_areas(piece, tokens) or last         # 去掉设备名再找，设备名里带的地点词不算用户说了地点
        bound.append(last)
    bound = [b or mentioned_areas(pieces[-1], tokens) for b in bound]   # 地点说在最后的（"……关了，甲地的"）
    # 都是完整的名字不管，除非地点只说在其中一台前面（"把书房灯和乙地的门灯关了"）：listed_place_conflict 只看这句说了乙地，会把书房灯拦下
    if all(m["rows"] == m["exact"] for m in found) and (all(bound) or not any(bound)):
        return [], None, False
    # 地点和位置也不算动作（"把甲地的客厅灯和甲地卧室的灯关了"）
    drop = sorted(({_squash(w) for a in set().union(*bound) for w in tokens[a]} | {_squash(p[0]) for p in rows}) - {""}, key=len, reverse=True)
    pieces = [re.sub("|".join(map(re.escape, drop)) or "$^", "", x) for x in pieces]
    if pieces[0].endswith(("把", "将")):
        pieces[0] = ""
    acts = _action_words(pieces)
    if acts is None or not (acts or asked):     # 有不是动作的话，或者一个动作都没说（答回问的话可以没有："甲地的客厅灯和卧室灯"）
        return [], None, False
    differ = len(set(acts)) >= 2
    if answering:
        answered = mentioned_areas(text, tokens)
        if not answered and not says_all(text):
            return [], _ask_place(found), False
        targets = []
        for m in found:
            r = [s for s in m["rows"] if s["house"] in answered] if answered else m["rows"]
            r = ([s for s in r if s in m["exact"]] or r) if answered else r
            if answered and len(r) > 1:         # 答了地点，那里还是有好几盏对得上：不替用户挑
                return [], _ask_place(found), False
            targets += [s["id"] for s in r]
        if not targets:
            return [], _ask_place(found), False
        return list(dict.fromkeys(targets)), None, differ
    if len(found) == 1:                         # 说了"都"：那一处对得上的全算，几处都有就问
        r = [s for s in found[0]["rows"] if not bound[0] or s["house"] in bound[0]]
        if len({s["house"] for s in r}) > 1:
            return [], _ask_place(found), False
        settled = [s["id"] for s in r]
    else:
        settled = _settle(found, bound)
        if settled is None:
            return [], _ask_place(found), False
    targets = list(dict.fromkeys(settled))
    if len(targets) < 2 or (not differ and len({t.split(".")[0] for t in targets}) != 1):
        return [], None, False
    return targets, None, differ
