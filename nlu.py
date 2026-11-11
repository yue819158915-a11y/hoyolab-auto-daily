# -*- coding: utf-8 -*-
"""LLM 意图识别（智谱 GLM）+ 本地规则降级"""
import json
import os
import re

import requests

from . import store
from . import synonyms

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")


def _ca_bundle():
    """沙箱内 pyenv Python 可能缺少 CA 证书，优先用系统证书包"""
    for p in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt"):
        if os.path.exists(p):
            return p
    try:
        import certifi
        return certifi.where()
    except ImportError:
        return None


_CA_BUNDLE = _ca_bundle()


def load_config():
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def resolve_key(header_key=None):
    """Key 优先级：请求头 > 环境变量 > config.json

    环境变量用于部署场景：镜像里不落 Key，由平台注入 LLM_KEY。
    """
    if header_key and header_key.strip():
        return header_key.strip()
    for env in ("LLM_KEY", "ZHIPU_API_KEY"):
        v = os.environ.get(env)
        if v and v.strip():
            return v.strip()
    cfg = load_config()
    return cfg.get("llm_key") or None


SYSTEM_PROMPT = """你是广州美食查询解析器，服务广州本地食客的觅食助手。把用户的话解析为 JSON。

【mode 字段（必填）】
- "filter"：用户在表达觅食需求或修改筛选条件（含"再便宜点的""换一家火锅"这类追问）
- "chat"：用户在闲聊、追问推荐理由/店铺细节、或提出与筛选无关的问题（如"为什么推荐这家""这些店远吗""你刚说的那家怎么去"）
- 【对比/评价/选择类问题一律 "chat"】：出现"哪家好""哪个更适合""怎么选""值不值得""为什么推荐它""带爸妈去行不行""有没有停车位"等，即使句子里出现菜系或地点词，也判为 "chat"，不要重新筛选

【当 mode="filter" 时填写以下字段，mode="chat" 时其余字段全为 null/[]】
可用类目（type 只能取这些值或 null）：粤菜、早茶、火锅、海鲜、甜品、小吃、烧烤宵夜、日料、异国风味、茶餐厅
⚠️ 不确定归哪一类就填 null，**不要填「其他」**：它是垃圾桶类目，库里只有 3 家互不相关的店，填了会把筛选范围限死到这几家。
常见地点（location 填原文地名或 null）：西华路、宝华路、北京路、上下九、陈家祠、淘金、珠江新城、五羊邨、石牌桥、天河北、南村万博、市桥、琶洲、滨江中、滨江东、东晓路、客村、江南西、新造、长堤、顺德（区域）、东莞（区域）、番禺（区域）、连锁

理解规则：
- 口语/方言识别：饮茶=早茶，打边炉=火锅，食海鲜=海鲜，糖水=甜品，宵夜=烧烤宵夜，撸串=烧烤宵夜
- 【词形用短词】业态词要和收录数据里的写法对得上：自助餐 / 放题 / 涮涮锅 一律写「自助」；数据里是「自助涮涮锅」「全牛自助牛肉火锅」这种写法，只有「自助」能全部匹配
- 价格："人均不超过X / X以内 / X块以下" → price_max = X；"人均X / X左右 / X上下" → price_min = X 的 70%，price_max = X 的 115%（如人均40 → 28~46）；"以上/起码X" → price_min = X；"便宜/实惠/平靓正/学生党" → price_max 设 50~60；"贵/高档/请客/商务" → price_min 设 150~200
- 【人群与口味偏好转成关键词】不要把人群词本身放进 keywords，翻成数据里大概率出现过的具体菜品：
  * 适合老人家/长辈/父母 → keywords 加 "炖汤""粥""白切鸡"，type 倾向 粤菜
  * 小孩/亲子 → keywords 加 "点心""甜品"
  * 养生/健康 → keywords 加 "炖汤""粥"
  * 怕辣/不吃辣 → type 倾向 粤菜/早茶/甜品
  * 想吃硬菜/大餐 → type=粤菜，price_min 设 100
  * ⚠️ 只用具体菜品或菜系名，**不要臆造「清淡」「锅气」「氛围」「安静」这类形容词**——收录数据里没有这些词，会导致直接查不到
- keywords 是「或」关系（命中任一即可），所以可以给 2~4 个候选词，不必互相收窄
- "最便宜/性价比" → sort="cheap"；"人气高/最火" → sort="popular"；"贵的那档" → sort="expensive"
- 指代消解："再便宜点的""换一家"要结合历史条件在上一轮基础上修改

输出 JSON（只输出 JSON，不要多余文字）：
{"mode":"filter"|"chat","type":string|null,"keywords":string[],"price_min":int|null,"price_max":int|null,"location":string|null,"sort":"cheap"|"expensive"|"popular"|null}
没有对应条件就填 null。"""

CATEGORIES = store.CATEGORIES
_SORTS = {"cheap", "expensive", "popular"}


# 关键词词形归一：模型/用户的口语说法 -> 数据里真实存在的写法。
# 只收录"已确认关键词匹配不到"的变体，发现新的就往这里加一行。
KEYWORD_ALIASES = {
    "自助餐": "自助", "自助火锅": "自助", "放题": "自助", "涮涮锅": "自助",
}


def _canon(kw: str) -> str:
    return KEYWORD_ALIASES.get(kw, kw)


def _validate(raw: dict):
    """校验并清洗 LLM 输出，非法值置 null"""
    def as_int(v):
        try:
            n = int(v)
            return n if 0 <= n <= 2000 else None
        except (TypeError, ValueError):
            return None

    mode = raw.get("mode")
    if mode not in ("filter", "chat"):
        mode = "filter"
    typ = raw.get("type")
    if typ == "其他":
        # 「其他」是垃圾桶类目：库里只有 3 家互不相关的店（鲁菜/川菜/自助）。
        # 用它做硬过滤会把结果限死，而放宽阶梯里类目永不放宽，等于死锁。
        # 视为「未指定」，让价格与关键词正常发挥作用。
        typ = None
    elif typ not in CATEGORIES:
        typ = None
    kws = raw.get("keywords")
    if not isinstance(kws, list):
        kws = []
    else:
        kws = [_canon(str(k).strip()) for k in kws if k and str(k).strip()]
        seen, uniq = set(), []
        for k in kws:
            if k not in seen:
                seen.add(k)
                uniq.append(k)
        kws = uniq[:6]
    loc = raw.get("location")
    if not isinstance(loc, str) or not loc.strip() or len(loc) > 12:
        loc = None
    sort = raw.get("sort")
    if sort not in _SORTS:
        sort = None
    return {
        "mode": mode,
        "category": typ, "keywords": kws,
        "price_min": as_int(raw.get("price_min")), "price_max": as_int(raw.get("price_max")),
        "location": loc, "sort": sort,
    }


def call_llm(messages, key, base_url, model, temperature=0.1, json_mode=True, timeout=10):
    body = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        json=body, timeout=timeout,
        verify=_CA_BUNDLE if _CA_BUNDLE else True)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_intent(message: str, history=None, header_key=None):
    """返回 (conditions, source)。source: llm | rule"""
    cfg = load_config()
    key = resolve_key(header_key)
    prev = _last_conditions(history)
    if not key:
        conditions = synonyms.parse_local(message)
        conditions["mode"] = "filter"
        return _inherit(conditions, prev), "rule"

    base_url = cfg.get("llm_base_url", "https://open.bigmodel.cn/api/paas/v4")
    model = cfg.get("llm_model", "glm-4-flash")

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (history or [])[-4:]:
        role = "user" if h.get("role") != "assistant" else "assistant"
        content = h.get("conditions_json") if role == "assistant" else h.get("text")
        if content:
            msgs.append({"role": role, "content": str(content)})
    msgs.append({"role": "user", "content": message})

    for attempt in range(2):
        try:
            content = call_llm(msgs, key, base_url, model, json_mode=(attempt == 0), timeout=(8 if attempt == 0 else 6))
            text = content.strip()
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("not a json object")
            return _inherit(_validate(data), prev), "llm"
        except Exception:
            continue
    conditions = synonyms.parse_local(message)
    conditions["mode"] = "filter"
    return _inherit(conditions, prev), "rule"


def _last_conditions(history):
    """取最近一轮 assistant 的条件 JSON"""
    for h in reversed(history or []):
        if h.get("role") == "assistant" and h.get("conditions_json"):
            try:
                c = json.loads(h["conditions_json"])
                return c if isinstance(c, dict) else None
            except (ValueError, TypeError):
                return None
    return None


def _inherit(conditions, prev):
    """指代消解兜底：本轮为筛选且未提及类目/地点时，继承上一轮的类目/地点"""
    if not prev or conditions.get("mode") != "filter":
        return conditions
    if conditions.get("category") is None and not conditions.get("keywords"):
        conditions["category"] = prev.get("category")
    if conditions.get("location") is None and not conditions.get("keywords"):
        conditions["location"] = prev.get("location")
    return conditions


# ============================ 做菜（cook）模式 ============================

# 做菜意图识别：动词类（做/煮/炒）+ 场景类（几人/分钟/清淡/忌口）+ 营养类
COOK_HINTS = re.compile(
    r"(做菜|做饭|做一|做点|煮|炒|蒸|炖|烤|炸|凉拌|焖|煲|菜谱|食谱|做法|怎么做|怎么烧|"
    r"热量|卡路里|大卡|kcal|营养|减脂|增肌|控糖|高蛋白|低碳|蛋白质|碳水|脂肪|"
    r"几人份|一个人吃|两个人吃|三个人吃|几个人吃|带饭|便当|早餐吃|晚餐做|宵夜做|"
    r"过敏|忌口|不吃|清淡|低卡|快手菜|家常菜|半成品|"
    r"换一批|换一道|换个菜|换个口味|换换口味|别的菜|其他的菜|其他菜|还有别的|再来一道|还有吗)")

# 强信号：出现即判定为"做饭"
COOK_STRONG = re.compile(
    r"(做法|菜谱|食谱|怎么做|怎么烧|煮|炒|蒸|炖|烤|炸|凉拌|焖|煲|做饭|做菜|做一|做点|"
    r"热量|卡路里|大卡|kcal|营养|减脂|增肌|控糖|高蛋白|低碳|几人份|几个人吃|带饭|便当|快手菜)")
# 弱信号：单独出现不足以判定（要配合人数/时间，且没有明显的"找店"特征）
COOK_SOFT = re.compile(
    r"(清淡|忌口|过敏|不吃|低卡|家常菜|换一批|换一道|换个菜|换个口味|换换口味|"
    r"别的菜|其他的菜|其他菜|还有别的|再来一道|还有吗)")
# 找店特征
EAT_HINT = re.compile(r"(人均|元以内|块钱|块以内|店|餐厅|馆子|哪家|推荐几|附近|外卖|喝茶|饮茶|早茶|宵夜吃)")


def is_cook_intent(msg):
    """分层判定：强信号直接做菜；弱信号要有人数/时间且不带找店特征。"""
    if COOK_STRONG.search(msg):
        return True
    if COOK_SOFT.search(msg):
        if EAT_HINT.search(msg) and re.search(r"(人均|元以内|块钱)", msg):
            return False
        if re.search(r"(\d+\s*分钟|\d+\s*个?人|两|二|三|四|五|六)\s*人", msg) or "一个人" in msg or "两个人" in msg:
            return True
        if not EAT_HINT.search(msg):
            return True
    return False


COOK_PROMPT = """你是家庭烹饪需求解析器。把用户的话解析为 JSON 约束。

字段说明：
- people: 用餐人数（整数，1~10），没提就 null
- max_time: 可用烹饪时间（分钟，整数，如"半小时"=30、"快手"=20），没提就 null
- budget: 人均预算（元，整数），没提就 null
- avoid: 忌口与过敏原数组（如 ["香菜","海鲜","花生"]），没提就 []
- goal: "减脂" | "增肌" | "控糖" | "均衡"，没提就 null
- meal: "breakfast" | "lunch" | "teatime" | "dinner" | "supper"，没提就 null
- cuisine: "粤式" | "家常" | "川味" | null
- keywords: 想吃的菜名或食材（如 ["番茄","鸡蛋","鸡胸肉"]），最多 3 个
- existing: 现有食材数组（用户说"我只有…""冰箱里有…"）

只输出 JSON：
{"people":int|null,"max_time":int|null,"budget":int|null,"avoid":[],"goal":null,"meal":null,"cuisine":null,"keywords":[],"existing":[]}
没有任何约束就全部填 null/[]。"""


def parse_cook_intent(message, history=None, header_key=None):
    """返回 (constraints, source)。source: llm | rule"""
    cfg = load_config()
    key = resolve_key(header_key)
    prev = _last_conditions(history)
    base = _rule_cook(message)
    if prev and prev.get("mode") == "cook":
        for k in ("people", "max_time", "budget", "goal", "avoid"):
            if not base.get(k) and prev.get(k):
                base[k] = prev[k]

    if not key:
        return base, "rule"

    msgs = [{"role": "system", "content": COOK_PROMPT}]
    for h in (history or [])[-4:]:
        role = "user" if h.get("role") != "assistant" else "assistant"
        content = h.get("conditions_json") if role == "assistant" else h.get("text")
        if content:
            msgs.append({"role": role, "content": str(content)})
    msgs.append({"role": "user", "content": message})

    try:
        content = call_llm(msgs, key, cfg.get("llm_base_url", "https://open.bigmodel.cn/api/paas/v4"),
                           cfg.get("llm_model", "glm-4-flash"), json_mode=True, timeout=8)
        text = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
        data = json.loads(text)
        out = {
            "mode": "cook",
            "people": _as_int(data.get("people"), 0, 10),
            "max_time": _as_int(data.get("max_time"), 5, 240),
            "budget": _as_int(data.get("budget"), 1, 500),
            "avoid": [str(x) for x in (data.get("avoid") or [])][:6],
            "goal": data.get("goal") if data.get("goal") in ("减脂", "增肌", "控糖", "均衡") else None,
            "meal": data.get("meal") if data.get("meal") in ("breakfast", "lunch", "teatime", "dinner", "supper") else None,
            "cuisine": data.get("cuisine") if data.get("cuisine") in ("粤式", "家常", "川味") else None,
            "keywords": [str(x) for x in (data.get("keywords") or [])][:3],
            "existing": [str(x) for x in (data.get("existing") or [])][:8],
        }
        return out, "llm"
    except Exception:
        return base, "rule"


def _as_int(v, lo, hi):
    try:
        n = int(v)
        return n if lo <= n <= hi else None
    except (TypeError, ValueError):
        return None


def _rule_cook(msg):
    """本地规则兜底：从文本里抠人数/时间/预算/目标"""
    c = {"mode": "cook", "people": None, "max_time": None, "budget": None,
         "avoid": [], "goal": None, "meal": None, "cuisine": None,
         "keywords": [], "existing": []}
    m = re.search(r"(\d+|两|二|三|四|五|六)\s*(?:个)?人", msg)
    if m:
        num = {"两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6}.get(m.group(1)) or int(m.group(1)) if m.group(1).isdigit() or m.group(1) in "两二三四五六" else None
        c["people"] = num
    m = re.search(r"(\d+)\s*(?:分钟|min)", msg) or re.search(r"半小时", msg)
    if m:
        c["max_time"] = 30 if m.group(0) == "半小时" else int(m.group(1))
    if "半小时" in msg:
        c["max_time"] = 30
    if "一小时" in msg:
        c["max_time"] = 60
    m = re.search(r"(\d+)\s*(?:块|元|蚊)", msg)
    if m:
        c["budget"] = int(m.group(1))
    for g in ("减脂", "增肌", "控糖", "均衡"):
        if g in msg:
            c["goal"] = g
    for meal, keys in (("breakfast", ("早餐",)), ("lunch", ("午餐", "中午")),
                       ("dinner", ("晚餐", "晚上")), ("supper", ("宵夜", "夜宵"))):
        if any(k in msg for k in keys):
            c["meal"] = meal
    for a in ("花生", "坚果", "海鲜", "虾", "蟹", "蛋", "奶", "香菜", "辣", "牛", "猪"):
        if f"不吃{a}" in msg or f"{a}过敏" in msg:
            c["avoid"].append(a)
    return c
