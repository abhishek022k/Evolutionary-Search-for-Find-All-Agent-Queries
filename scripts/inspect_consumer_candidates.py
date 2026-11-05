"""Read-only, free check of consumer-behaviour questions in Item 1A + Item 7.
`signature` = the catchy words a simple keyword search would try (any location).
`near` = those words within 300 characters of COVID language.
Word counts only; no labels. Writes nothing."""
import glob
import json
import re

COVID = re.compile(r"covid|coronavirus|pandemic", re.I)
SPAN = 300

CANDIDATES = {
    "panic buying / stockpiling": (
        r"\b(stockpil\w*|stock(?:ing|ed)?[- ]up|pantry[- ]?(?:load\w*|stock\w*)|"
        r"panic[- ](?:buy\w*|purchas\w*)|hoard\w*|bulk[- ]buying)\b"),
    "stay-at-home hobby or lifestyle boom": (
        r"\b(stay(?:ing)?[- ]at[- ]home|at[- ]home (?:activit\w*|entertainment|"
        r"fitness|workout\w*|cooking|baking|consumption|dining)|more time at home|"
        r"home[- ]?bound|cocoon\w*|nesting|home improvement|gardening|"
        r"outdoor (?:activit\w*|recreation)|socially distan\w* activit\w*|"
        r"do[- ]it[- ]yourself|DIY)\b"),
    "fewer shopping trips but bigger baskets": (
        r"\b(fewer|less frequent|reduced|lower|decline in|decrease in)\s+(?:\w+\s+){0,3}?"
        r"(trips?|visits?|traffic|transactions)\b[^.]{0,160}\b(larger|bigger|higher|"
        r"increase\w* in)\s+(?:\w+\s+){0,3}?(baskets?|tickets?|transaction (?:size|value)s?|"
        r"average (?:ticket|check|order))\b|\b(larger|bigger|higher)\s+(?:\w+\s+){0,3}?"
        r"(baskets?|tickets?|average check|average order)\b[^.]{0,160}\b(fewer|lower|"
        r"less frequent|reduced|decline\w*)\s+(?:\w+\s+){0,3}?(trips?|visits?|traffic|"
        r"transactions)\b"),
    "stimulus checks changed spending": (
        r"\b(stimulus (?:check|payment|money|fund)\w*|economic impact payments?|"
        r"government stimulus|enhanced unemployment|unemployment benefits)\b"),
    "eating at home replaced eating out": (
        r"\b(food[- ]at[- ]home|at[- ]home (?:consumption|dining|eating|meals?|cooking)|"
        r"away[- ]from[- ]home|retail grocery|shift\w* (?:in|to|toward)[^.]{0,40}"
        r"(?:grocery|home consumption))\b"),
    "shift to contactless or mobile payment": (
        r"\b(contactless|touchless|tap[- ]to[- ]pay|mobile (?:order\w*|pay\w*)|"
        r"digital wallet\w*|curbside)\b"),
}
COMPILED = {k: re.compile(v, re.I) for k, v in CANDIDATES.items()}

n = 0
stats = {k: dict(sig=0, near7=0, near1a=0, near=0) for k in CANDIDATES}


def near_covid(text, rx):
    for m in rx.finditer(text):
        lo, hi = max(0, m.start() - SPAN), m.end() + SPAN
        if COVID.search(text, lo, hi):
            return True
    return False


for path in sorted(glob.glob("data/raw/2020_*.jsonl")):
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            a = r.get("section_1A") or ""
            if len(a.split()) < 250:
                continue
            s7 = r.get("section_7") or ""
            n += 1
            for k, rx in COMPILED.items():
                st = stats[k]
                st["sig"] += bool(rx.search(a) or rx.search(s7))
                in7, in1a = near_covid(s7, rx), near_covid(a, rx)
                st["near7"] += in7
                st["near1a"] += in1a
                st["near"] += in7 or in1a

print("filings:", n)
for k, st in stats.items():
    print(f"{k}\n   signature words anywhere: {st['sig']} ({100 * st['sig'] / n:.1f}%)"
          f" | near COVID: {st['near']} ({100 * st['near'] / n:.1f}%)"
          f" | Item 7: {st['near7']} | Item 1A: {st['near1a']}")
