"""Read-only, free check of candidate questions for Item 7 (and Item 1A).
`anchor` = the obvious key word, so its share shows how much of a giveaway it is.
`near` = rough concept language close to COVID, or close to the anchor for the
non-COVID question. Word counts only; no labels. Writes nothing."""
import glob
import json
import re

COVID = re.compile(r"covid|coronavirus|pandemic", re.I)
SPAN = 300

CANDIDATES = {
    "new product/service launched because of COVID": dict(
        anchor=r"\b(launch\w*|introduc\w*|new product\w*|new service\w*|began (?:offering|selling|producing|manufacturing))\b",
        near=r"\b(launched|introduced|began (?:offering|selling|producing|manufacturing)|started (?:offering|selling|producing|making)|pivot\w*|repurpos\w*|retool\w*|new (?:product|service|offering)s?)\b",
        covid=True),
    "dividend or buyback paused because of COVID": dict(
        anchor=r"\b(dividends?|repurchase\w*|buybacks?)\b",
        near=r"\b(suspend\w*|paus\w*|halt\w*|discontinu\w*|eliminat\w*|reduc\w*|ceas\w*)\b[^.]{0,80}\b(dividends?|repurchase\w*|buybacks?)\b|\b(dividends?|repurchase\w*|buybacks?)\b[^.]{0,80}\b(suspend\w*|paus\w*|halt\w*|discontinu\w*|eliminat\w*|ceas\w*)\b",
        covid=True),
    "debt covenant waived or amended because of COVID": dict(
        anchor=r"\bcovenants?\b",
        near=r"\b(waiv\w*|amend\w*|relief|holiday|suspend\w*)\b[^.]{0,100}\bcovenants?\b|\bcovenants?\b[^.]{0,100}\b(waiv\w*|amend\w*|relief|holiday|suspend\w*)\b",
        covid=True),
    "executive pay cut because of COVID": dict(
        anchor=r"\b(salar\w*|executive compensation|base pay)\b",
        near=r"\b(reduc\w*|cut\w*|forgo\w*|forego\w*|defer\w*|waiv\w*)\b[^.]{0,80}\b(salar\w*|base pay|executive compensation)\b|\b(salar\w*)\b[^.]{0,60}\b(reduc\w*|cut\w*)\b",
        covid=True),
    "rent relief from landlords because of COVID": dict(
        anchor=r"\b(rents?|leases?|landlords?)\b",
        near=r"\b(rent|lease|landlord\w*)\b[^.]{0,80}\b(deferr\w*|concession\w*|abatement\w*|forgiv\w*|relief)\b|\b(deferr\w*|concession\w*|abatement\w*)\b[^.]{0,80}\b(rent|lease|landlord\w*)\b",
        covid=True),
    "revenue fell because a major customer was lost": dict(
        anchor=r"\bcustomers?\b",
        near=r"\b(loss of|lost|departure of|termination (?:of|by)|non-?renewal)\b[^.]{0,60}\b(largest|major|significant|key|principal)\s+customer\w*\b|\b(largest|major|significant|key|principal)\s+customer\w*\b[^.]{0,80}\b(terminat\w*|lost|ceased|did not renew|reduced (?:its )?orders)\b",
        covid=False),
}
for c in CANDIDATES.values():
    c["anchor_re"] = re.compile(c["anchor"], re.I)
    c["near_re"] = re.compile(c["near"], re.I)

n = 0
stats = {k: dict(anchor=0, near_7=0, near_1a=0, near_any=0) for k in CANDIDATES}


def near_hit(text, cfg):
    if not cfg["covid"]:
        return bool(cfg["near_re"].search(text))
    for m in COVID.finditer(text):
        lo, hi = max(0, m.start() - SPAN), m.end() + SPAN
        if cfg["near_re"].search(text, lo, hi):
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
            for k, cfg in CANDIDATES.items():
                st = stats[k]
                st["anchor"] += bool(cfg["anchor_re"].search(a) or cfg["anchor_re"].search(s7))
                in7, in1a = near_hit(s7, cfg), near_hit(a, cfg)
                st["near_7"] += in7
                st["near_1a"] += in1a
                st["near_any"] += in7 or in1a

print("filings:", n)
for k, st in stats.items():
    print(f"{k}\n   anchor word in 1A or 7: {100 * st['anchor'] / n:.1f}%"
          f" | rough hits Item 7: {st['near_7']} ({100 * st['near_7'] / n:.1f}%)"
          f" | Item 1A: {st['near_1a']} ({100 * st['near_1a'] / n:.1f}%)"
          f" | either: {st['near_any']} ({100 * st['near_any'] / n:.1f}%)")
