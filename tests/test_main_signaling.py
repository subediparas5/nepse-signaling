import re

import main_signaling as ms


def test_llm_skipped_without_api_key(monkeypatch):
    monkeypatch.setattr(ms, "OPEN_AI_API_KEY", None)

    def boom(*a, **k):
        raise AssertionError("OpenAI client must not be constructed without a key")

    monkeypatch.setattr(ms, "OpenAI", boom)
    assert ms.get_llm_picks([{"symbol": "NABIL"}]) == ""


def test_digest_omits_llm_section_when_empty():
    body = ms.format_telegram_digest("", [], [])
    assert "<b>LLM</b>" not in body
    assert "<b>Strict BUY</b> (0)" in body


def test_digest_renders_llm_pipe_table():
    body = ms.format_telegram_digest("NABIL | Rs 500 | good\nNICA | Rs 400 | ok", [], [])
    assert "<b>LLM</b>" in body
    assert "<pre>" in body and "NABIL" in body


def test_near_52w_low_rules():
    base = {"ltp": 52, "week_52_high": 150, "week_52_low": 50}
    assert ms._near_52w_low(base)
    assert not ms._near_52w_low({**base, "ltp": 100})
    assert not ms._near_52w_low({**base, "eps_ttm": -1})
    assert not ms._near_52w_low({**base, "ma120": 0})


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


def test_persist_daily_snapshot_writes_prices_signals_and_securities(tmp_path):
    import history_store as hs
    from nepse_signal_rules import classify_nepse_signal

    stock = {
        "symbol": "NABIL", "sector": "BANKING", "ltp": 569.0, "open": 566.0, "high": 570.0, "low": 565.0,
        "vwap": 568.2, "prev_close": 565.0, "volume": 70749.0, "turnover": 40205031.3, "transactions": 495.0,
        "week_52_high": 620.0, "week_52_low": 480.0,
    }
    stock.update(classify_nepse_signal(stock, "BANKING"))
    listed = [{"symbol": "NABIL", "security_id": 131, "sector": "BANKING"}]
    ms.persist_daily_snapshot("2026-09-24", [stock], listed, data_dir=tmp_path)

    [price] = hs.load_prices(tmp_path)
    assert (price["close"], price["open"], price["trades"], price["week_52_low"]) == ("569", "566", "495", "480")
    [sig] = hs._read(tmp_path / "signals" / "2026-09.csv")
    assert sig["verdict"] == stock["signal_verdict"] and sig["close"] == "569"
    assert hs._read(tmp_path / "securities.csv")[0]["security_id"] == "131"
