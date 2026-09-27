"""
NEPSE exchange notices (NOTS news feed) as dated, per-symbol events.

Each notice is typed with keyword rules (English + Nepali) over its title and body. A zero-shot
Laya test agreed with these on ~88% of notices and was worse on the rest (IPO vs bonus listings,
Nepali trading halts, transaction releases), so rules are the labeller. Most bodies only say
"see the attached notice", so the title carries nearly all the information.

Timing: notices are posted after the session (typically ~15:40 NPT). An event dated D is known
at D's close, so a backtest may act on it from the next session onward.
"""

from __future__ import annotations

import re
from typing import Any

EVENT_TYPES = {
    "price_adjustment": "share price adjusted for bonus shares, right shares or cash dividend (ex-date next session)",
    "bonus_listing": "listing of newly issued bonus shares",
    "right_listing": "listing of right shares or auctioned promoter/right shares",
    "ipo_listing": "listing of a new company's IPO or FPO shares",
    "debt_listing": "listing of a debenture, government bond or preference share",
    "fund_listing": "listing of mutual fund units",
    "dividend_book_closure": "dividend declaration or book closure notice",
    "trading_halt": "trading of a security halted or suspended",
    "trading_resume": "trading of a security released, resumed or unfrozen",
    "promoter_conversion": "conversion of promoter shares to ordinary shares",
    "merger": "merger or acquisition between companies, or listing of the merged entity",
    "other": "anything else, such as delisting, fees, staff recruitment or system notices",
}

_UNIT = re.compile(r"\bunits?\b")


def classify_notice(title: str, text: str) -> str:
    t = f"{title} {text}".lower()

    def has(*words: str) -> bool:
        return any(w in t for w in words)

    if has("price adjust", "adjusted price"):
        return "price_adjustment"
    if has("रोक्का", "halt", "suspen", "बन्द गरिएको"):
        return "trading_halt"
    if has("transactions release", "transaction release", "फुकुवा", "resum"):
        return "trading_resume"
    if has("conversion of promoter", "conversion promoter", "promoter share conversion"):
        return "promoter_conversion"
    listing = ("listing" in t and not has("de-listing", "delisting")) or has("सूचीकृत", "सूचीकरण")
    if listing:
        if has("after merger", "merged", "मर्जर"):
            return "merger"
        if "bonus" in t:
            return "bonus_listing"
        if "right" in t:
            return "right_listing"
        if has("ipo", "fpo", "initial public", "further public"):
            return "ipo_listing"
        if has("debenture", "bond", "rinpatra", "ऋणपत्र", "preference", "अग्राधिकार"):
            return "debt_listing"
        if _UNIT.search(t) or has("fund", "scheme", "yojana", "samriddhi", "horizon", "योजना"):
            return "fund_listing"
    if has("dividend", "book clos", "बुक क्लोज", "लाभांश"):
        return "dividend_book_closure"
    if has("merger", "मर्जर", "गाभ", "acquisition", "प्राप्ति"):
        return "merger"
    return "other"


_SYM = r"[A-Z][A-Z0-9]{1,9}"
_JOIN = r"\s*(?:&|,|and)\s*"
_PAREN = re.compile(rf"\(\s*({_SYM}(?:{_JOIN}{_SYM})*)\s*\)")
_TAIL = re.compile(rf"[-–]\s*\(?\s*({_SYM}(?:{_JOIN}{_SYM})*)\s*\)?\s*$")
_OF_TAIL = re.compile(rf"\bof\s+({_SYM}(?:{_JOIN}{_SYM})*)\s*$")
_BODY_OF = re.compile(rf"\bof\s+({_SYM})\s+is\b")
_NOT_SYMBOLS = {"NEPSE", "IPO", "FPO", "QII", "QIIS", "AGM", "SGM", "NPR", "RS"}


def _split(group: str) -> list[str]:
    return [s for s in re.split(_JOIN, group.strip()) if s]


def notice_symbols(title: str, text: str) -> list[str]:
    """Symbols a notice refers to: '(SYM)', '(A & B)', '- SYM' or 'of A & B' in the title, else the body."""
    title = (title or "").strip()
    found: list[str] = []
    for pattern in (_PAREN, _TAIL, _OF_TAIL):
        for g in pattern.findall(title):
            found.extend(_split(g))
        if found:
            break
    if not found:
        for g in _PAREN.findall(text or "") + _BODY_OF.findall(text or ""):
            found.extend(_split(g))
    out: list[str] = []
    for s in found:
        s = s.upper()
        if s not in _NOT_SYMBOLS and s not in out:
            out.append(s)
    return out


def notices_to_events(notices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One event row per (notice, symbol). `notices` rows need id, date (YYYY-MM-DD), title, text."""
    events = []
    for n in notices:
        if not n.get("date"):
            continue
        etype = classify_notice(n.get("title", ""), n.get("text", ""))
        for sym in notice_symbols(n.get("title", ""), n.get("text", "")):
            events.append({
                "date": n["date"], "symbol": sym, "event": etype,
                "alert_id": n.get("id"), "title": " ".join(str(n.get("title", "")).split())[:200],
            })
    return events


def fetch_events() -> list[dict[str, Any]]:
    """Current NOTS news feed as event rows (network)."""
    from nepse_official import _client, _html_text

    notices = [
        {"id": a.get("id"), "date": (a.get("addedDate") or "")[:10], "title": a.get("messageTitle") or "",
         "text": _html_text(a.get("messageBody"))}
        for a in _client().get_news_alerts(use_cache=False) or []
    ]
    return notices_to_events(notices)
