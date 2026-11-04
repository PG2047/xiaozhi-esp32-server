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


# 用户明说要控制一类设备的词。"全"只认"全都 / 全部"和后面紧跟动作的（全关、全打开）：
# 11-03 外放验收里"除甲醛"被转写成"出价全"，光认"全"会把这种转写错的设备名当成"全部"
_ALL_WORDS = re.compile(r"都|全部|所有|全都|一起|统统|全(?=[关开打拉调停合])")
_NOT_ACTIONABLE = {"sensor", "binary_sensor"}


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
    if _ALL_WORDS.search(utterance or ""):
        return None
    hits = [e for e in matched(tool_name, a, entries) if e["domain"] not in _NOT_ACTIONABLE]
    if len(hits) <= 1:
        return None
    return ask_which(hits, f"{a['area']}" if isinstance(a.get("area"), str) else "家里")


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
    if not (utterance or "").strip() or _ALL_WORDS.search(utterance):
        return []
    hits = matched(tool_name, a, entries)
    if len(hits) != 1 or hits[0]["domain"] not in set(domains):
        return []
    e, text = hits[0], norm(recent).replace(" ", "")
    mine = _name_parts(e)
    group = [s for s in entries if s is e or (s["domain"] == e["domain"] and not any(x in text for x in mine - _name_parts(s)))]
    return group if len(group) > 1 else []


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
