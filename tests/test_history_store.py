import numpy as np
import pytest

import history_store as hs


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, ""), (569.0, "569"), (569.25, "569.25"), (1 / 3, "0.3333"), (7, "7"), (float("nan"), ""),
        (" A ", "A"), (np.float64(45.51189), "45.5119"), (np.int64(12), "12"), (np.float64("nan"), ""),
        (np.bool_(True), "1"),
    ],
)
def test_fmt_is_stable(value, expected):
    assert hs.fmt(value) == expected


def test_upsert_prices_is_idempotent_and_splits_by_month(tmp_path):
    rows = [
        {"date": "2026-08-31", "symbol": "NABIL", "close": 560.0, "volume": 1000.0},
        {"date": "2026-09-01", "symbol": "NABIL", "close": 565.5, "volume": 1200.0},
        {"date": "2026-09-01", "symbol": "ADBL", "close": 300.0},
    ]
    assert hs.upsert_prices(rows, tmp_path) == 3
    aug = (tmp_path / "prices" / "2026-08.csv").read_text()
    sep = (tmp_path / "prices" / "2026-09.csv").read_text()
    assert "2026-08-31,NABIL" in aug
    # Sorted by (date, symbol): ADBL before NABIL
    assert sep.index("ADBL") < sep.index("NABIL")

    assert hs.upsert_prices(rows, tmp_path) == 0
    assert (tmp_path / "prices" / "2026-09.csv").read_text() == sep


def test_empty_incoming_value_never_erases_stored_one(tmp_path):
    hs.upsert_prices([{"date": "2026-09-01", "symbol": "NABIL", "open": 560, "close": 565}], tmp_path)
    # History endpoint has no open; merging it must keep the snapshot's open.
    hs.upsert_prices([{"date": "2026-09-01", "symbol": "NABIL", "open": None, "close": 566, "trades": 400}], tmp_path)
    [row] = hs.load_prices(tmp_path)
    assert (row["open"], row["close"], row["trades"]) == ("560", "566", "400")


def test_rows_without_key_are_skipped(tmp_path):
    assert hs.upsert_prices([{"date": "2026-09-01", "close": 1}, {"symbol": "X", "close": 1}], tmp_path) == 0
    assert hs.load_prices(tmp_path) == []


def test_securities_keep_delisted_and_track_seen_range(tmp_path):
    hs.upsert_securities([{"symbol": "OLD", "security_id": 1, "sector": "BANKING"}], "2026-01-05", tmp_path)
    hs.upsert_securities([{"symbol": "NEW", "security_id": 2, "sector": "HYDROPOWER"}], "2026-09-24", tmp_path)
    hs.upsert_securities([{"symbol": "OLD", "security_id": 1, "sector": "BANKING"}], "2025-12-01", tmp_path)
    rows = {r["symbol"]: r for r in hs._read(tmp_path / "securities.csv")}
    assert set(rows) == {"OLD", "NEW"}
    assert (rows["OLD"]["first_seen"], rows["OLD"]["last_seen"]) == ("2025-12-01", "2026-01-05")


def test_signals_and_index_roundtrip(tmp_path):
    hs.upsert_index([{"date": "2026-09-24", "close": 2629.81, "pct_change": 0.44}], tmp_path)
    assert hs.load_index(tmp_path)[0]["close"] == "2629.81"
    hs.upsert_signals(
        [{"date": "2026-09-24", "symbol": "NABIL", "verdict": "BUY", "buy_score": 5, "reasons": "a | b, c"}],
        tmp_path,
    )
    [sig] = hs._read(tmp_path / "signals" / "2026-09.csv")
    assert (sig["verdict"], sig["buy_score"], sig["reasons"]) == ("BUY", "5", "a | b, c")
