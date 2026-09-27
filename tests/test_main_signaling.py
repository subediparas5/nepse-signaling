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
