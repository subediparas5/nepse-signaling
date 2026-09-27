import re

import pandas as pd

import main_signaling as ms


def test_llm_skipped_without_api_key(monkeypatch):
    monkeypatch.setattr(ms, "OPEN_AI_API_KEY", None)

    def boom(*a, **k):
        raise AssertionError("OpenAI client must not be constructed without a key")

    monkeypatch.setattr(ms, "OpenAI", boom)
    assert ms.get_llm_picks([{"symbol": "NABIL"}]) == ""


def test_compact_for_llm_rounds_and_drops_missing():
    out = ms.compact_for_llm([{"symbol": "NABIL", "score": 91.234567, "rsi14": None, "unrelated": 1}])
    assert out == [{"symbol": "NABIL", "score": 91.2346}]


def _setup(sym, cls, score, risk=40.0, ret=0.05):
    return {"symbol": sym, "ltp": 500.0, "classification": cls, "score": score, "risk_score": risk, "ret_20d": ret}


def test_select_setups_prefers_strong_then_score():
    stocks = [
        _setup("A", "SETUP", 99), _setup("B", "STRONG_SETUP", 91), _setup("C", "STRONG_SETUP", 97),
        _setup("D", "HIGH_RISK", 100), {"symbol": "E", "classification": "STRONG_SETUP", "score": None},
    ]
    assert [s["symbol"] for s in ms.select_setups(stocks, n=3)] == ["C", "B", "A"]


def test_attach_scores_marks_unscored_as_insufficient():
    scored = pd.DataFrame({"classification": ["SETUP"], "score": [80.0], "rsi14": [float("nan")]}, index=["NABIL"])
    stocks = [{"symbol": "NABIL"}, {"symbol": "NEWCO"}]
    ms.attach_scores(stocks, scored)
    assert stocks[0]["classification"] == "SETUP" and stocks[0]["score"] == 80.0
    assert stocks[0]["rsi14"] is None
    assert stocks[1]["classification"] == "INSUFFICIENT_DATA"


def test_digest_omits_llm_section_when_empty():
    body = ms.format_telegram_digest("", [], None, {})
    assert "LLM" not in body
    assert "<b>Top setups</b> (0 strong, 0 setup)" in body
    assert "Market regime unavailable" in body


def test_digest_renders_setups_regime_and_llm():
    ctx = {"regime": "BEARISH", "index_ret_20d": -0.031, "breadth_sma50": 0.42, "high_vol": True}
    body = ms.format_telegram_digest(
        "NABIL | Rs 500 | steady trend\nNICA | Rs 400 | ok",
        [_setup("NABIL", "STRONG_SETUP", 91.4), _setup("NICA", "SETUP", 80.2, ret=-0.012)],
        ctx,
        {"STRONG_SETUP": 1, "SETUP": 1, "HIGH_RISK": 3},
    )
    assert "Market: <b>BEARISH</b> · NEPSE 20d -3.1% · 42% above SMA50 · high volatility" in body
    assert "<b>LLM notes</b>" in body and "<pre>" in body
    assert "NABIL     500.0  91  40  +5.0  S+" in body
    assert "NICA      500.0  80  40  -1.2  S" in body
    assert "3 high risk" in body


def test_telegram_chunks_keep_pre_blocks_balanced():
    rows = "\n".join(f"ROW{i:04d} " + "x" * 60 for i in range(200))
    message = "<b>head</b>\n<pre>" + rows + "</pre>\ntail"
    chunks = ms._telegram_html_chunks(message, max_len=4096)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 4096
        assert c.count("<pre>") == c.count("</pre>")
    joined = "".join(chunks)
    assert all(f"ROW{i:04d}" in joined for i in range(200))
    assert re.search(r"tail$", chunks[-1])


def test_persist_snapshot_and_signals(tmp_path):
    import history_store as hs
    from nepse_signal_rules import classify_nepse_signal

    stock = {
        "symbol": "NABIL", "sector": "BANKING", "ltp": 569.0, "open": 566.0, "high": 570.0, "low": 565.0,
        "vwap": 568.2, "prev_close": 565.0, "volume": 70749.0, "turnover": 40205031.3, "transactions": 495.0,
        "week_52_high": 620.0, "week_52_low": 480.0,
    }
    stock.update(classify_nepse_signal(stock, "BANKING"))
    stock.update({"classification": "STRONG_SETUP", "score": 91.23456, "risk_score": 22.0})
    listed = [{"symbol": "NABIL", "security_id": 131, "sector": "BANKING"}]
    ms.persist_market_snapshot("2026-09-24", [stock], listed, data_dir=tmp_path)
    ms.persist_signals("2026-09-24", [stock], {"regime": "NEUTRAL"}, data_dir=tmp_path)

    [price] = hs.load_prices(tmp_path)
    assert (price["close"], price["open"], price["trades"], price["week_52_low"]) == ("569", "566", "495", "480")
    [sig] = hs._read(tmp_path / "signals" / "2026-09.csv")
    assert (sig["classification"], sig["score"], sig["regime"]) == ("STRONG_SETUP", "91.2346", "NEUTRAL")
    assert sig["verdict"] == stock["signal_verdict"] and sig["close"] == "569"
    assert hs._read(tmp_path / "securities.csv")[0]["security_id"] == "131"
