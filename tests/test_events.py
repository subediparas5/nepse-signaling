import pandas as pd
import pytest

import backtest as B
import events as E


@pytest.mark.parametrize("title, text, expected", [
    ("Price Adjusted - Everest Bank Limited (EBL)", "", "price_adjustment"),
    ("", "Adjusted Price of Jyoti Bikas Bank Limited is Rs.306.80 for 3% Bonus Shares", "price_adjustment"),
    ("Listing 10% Bonus Shares of Synergy Power Development (SPDL)", "", "bonus_listing"),
    ("Listing IPO Share of Appolo Hydropower Limited (APHL)", "", "ipo_listing"),
    ("Listing Shares of United Ajod Insurance Limited (UAIL) after merger", "", "merger"),
    ("Listing - MBL Equity Fund (MBLEF)", "", "fund_listing"),
    ("Listing Additional Unit ( Promoter of Nepal Government ) Share of Bishal Bazar", "", "fund_listing"),
    ("मर्जर प्रयोजनका लागि कारोबार रोक्का राखिएको - GIC & SGI", "", "trading_halt"),
    ("Transactions Release (Except Basic Shareholders) of CCBL", "", "trading_resume"),
    ("Conversion Promoter Shares of Mero Microfinance Bittiya Sanstha Ltd. (MERO)", "", "promoter_conversion"),
    ("Delisting of 10.25% SBL Debenture 2083 (SBLD83)", "", "other"),
    ("Book close notice of KEF, KDBY & KSY", "", "dividend_book_closure"),
])
def test_classify_notice(title, text, expected):
    assert E.classify_notice(title, text) == expected


def test_united_does_not_count_as_fund_units():
    assert E.classify_notice("Listing Shares of United Modi Hydropower (UMHL)", "") != "fund_listing"


@pytest.mark.parametrize("title, text, expected", [
    ("Price Adjusted - Everest Bank Limited (EBL)", "", ["EBL"]),
    ("मर्जर प्रयोजनका लागि धितोपत्रको कारोबार रोक्का राखिएको बारे - (PFL & SFCL)", "", ["PFL", "SFCL"]),
    ("Transactions Suspended of GBLBS & SAMAJ", "", ["GBLBS", "SAMAJ"]),
    ("Price Adjusted – NHDL", "", ["NHDL"]),
    ("", "Adjusted price of HATHY is 724.09 for 10% bonus shares", ["HATHY"]),
    ("Notice regarding annual fee payment.", "For more details find the attached NEPSE notice.", []),
    ("Registration of Qualified Institutional Investors (QIIs) in NEPSE", "", []),
])
def test_notice_symbols(title, text, expected):
    assert E.notice_symbols(title, text) == expected


def test_notices_to_events_one_row_per_symbol_and_skips_undated():
    rows = E.notices_to_events([
        {"id": 1, "date": "2026-01-05", "title": "Transactions Suspended of AAA & BBB", "text": ""},
        {"id": 2, "date": "", "title": "Price Adjusted - X (XYZ)", "text": ""},
    ])
    got = [(r["symbol"], r["event"], r["alert_id"]) for r in rows]
    assert got == [("AAA", "trading_halt", 1), ("BBB", "trading_halt", 1)]


def _scored(dates, syms):
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    n = len(idx)
    df = pd.DataFrame({"ret_20d": [0.0] * n, "fwd_mdd_20": [0.0] * n, "classification": ["NEUTRAL"] * n}, index=idx)
    for h in (1, 5, 10, 20):
        df[f"fwd_excess_{h}"] = [float(i) for i in range(n)]
    return df


def test_event_rows_use_last_session_on_or_before_notice_and_dedupe():
    dates = ["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
    df = _scored(dates, ["A"])
    events = pd.DataFrame([
        {"date": "2026-01-03", "symbol": "A", "event": "bonus_listing"},   # weekend -> session 01-02
        {"date": "2026-01-05", "symbol": "A", "event": "bonus_listing"},   # repeat within 5 sessions -> dropped
        {"date": "2026-01-05", "symbol": "A", "event": "trading_halt"},    # other type kept
        {"date": "2025-12-01", "symbol": "A", "event": "merger"},          # before history -> skipped
        {"date": "2026-01-05", "symbol": "ZZZ", "event": "merger"},        # unknown symbol -> skipped
    ])
    ev = B.event_rows(df, events)
    assert list(zip(ev["event"], ev["date"])) == [("bonus_listing", "2026-01-02"), ("trading_halt", "2026-01-05")]
    assert ev.iloc[0]["fwd_excess_20"] == 1.0  # the 01-02 row's value, never a later one


def test_event_report_marks_small_groups():
    ev = pd.DataFrame([{"event": "merger", "fwd_excess_1": 0, "fwd_excess_5": 0, "fwd_excess_10": 0,
                        "fwd_excess_20": 0.01, "fwd_mdd_20": 0, "pre_excess_20": 0, "classification": "SETUP"}])
    assert "Too few to analyse: merger (1)" in B.build_event_report(ev, ("a", "b"))
