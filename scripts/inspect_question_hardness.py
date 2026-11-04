"""Read-only, free check: how far would plain keywords get on the China/Taiwan
question? Counts words only; involves no labels. Writes nothing."""
import glob
import json
import re

DIRECT = re.compile(r"\b(china|chinese|prc|taiwan|taiwanese)\b", re.I)
# Places and firms that imply China/Taiwan without necessarily saying so.
INDIRECT = re.compile(
    r"\b(shenzhen|shanghai|guangdong|guangzhou|dongguan|suzhou|beijing|wuxi|"
    r"hsinchu|taipei|tsmc|foxconn|hon hai|umc|smic|pegatron|wistron|quanta|"
    r"compal|wuhan)\b", re.I)
VAGUE = re.compile(r"\b(asia|asian|asia[- ]pacific|far east|southeast asia)\b", re.I)
SUPPLY = re.compile(
    r"\b(supplier|suppliers|manufacturer|manufacturers|manufactured|foundry|"
    r"foundries|contract manufactur\w+|vendor|vendors|supply)\b", re.I)
SINGLE = re.compile(
    r"\b(sole|single|one|limited number|few|exclusive|sole[- ]source\w*|"
    r"single[- ]source\w*|primary|principal|depend\w*|rely|relies|reliance)\b", re.I)


def windows(text, size=600, step=300):
    for i in range(0, max(1, len(text) - size + 1), step):
        yield text[i:i + size]


n = 0
direct = taiwan_only_any = 0
indirect_no_direct = vague_no_direct = 0
china_near_supply = china_near_supply_single = 0
china_only_with_covid_context = 0

for path in sorted(glob.glob("data/raw/2020_*.jsonl")):
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            t = r.get("section_1A") or ""
            if len(t.split()) < 250:
                continue
            n += 1
            has_direct = bool(DIRECT.search(t))
            direct += has_direct
            if re.search(r"\b(taiwan|taiwanese)\b", t, re.I):
                taiwan_only_any += 1
            if not has_direct:
                indirect_no_direct += bool(INDIRECT.search(t))
                vague_no_direct += bool(VAGUE.search(t) and SUPPLY.search(t))
                continue
            near = both = False
            for w in windows(t):
                if DIRECT.search(w) and SUPPLY.search(w):
                    near = True
                    if SINGLE.search(w):
                        both = True
                        break
            china_near_supply += near
            china_near_supply_single += both
            if not near and re.search(r"covid|coronavirus", t, re.I):
                china_only_with_covid_context += 1


def pct(x):
    return f"{x} ({100 * x / n:.1f}%)"


print("filings with 250+ word Item 1A:", n)
print("mention China/Chinese/PRC/Taiwan:", pct(direct))
print("  of those, mention Taiwan:", pct(taiwan_only_any))
print("  China term within 600 chars of a supply word:", pct(china_near_supply))
print("  ...and also a single/dependence word nearby:", pct(china_near_supply_single))
print("  China term, never near supply, filing mentions COVID:",
      pct(china_only_with_covid_context))
print("no direct word, but a named city/firm (Shenzhen, TSMC...):", pct(indirect_no_direct))
print("no direct word, but Asia + supply words:", pct(vague_no_direct))
