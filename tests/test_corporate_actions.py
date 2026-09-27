import pytest

from nepse_official import parse_price_adjustment

CASES = [
    ("Price Adjusted - Kalinchowk Darshan Limited (KDL)",
     "<p><span>Adjusted price of Kalinchowk Darshan Limited (KDL) is Rs.580.46 for 8.50% Bonus shares on previous "
     "closing price of Rs.629.80</span></p>", "KDL", 580.46, 629.80),
    ("Price Adjusted – Life Insurance Corporation (Nepal) Limited (LICN)\t",
     "Adjusted price of Life Insurance Corporation (Nepal) Limited (LICN) is Rs.825.78 for 10% Bonus Shares on "
     "previous closing price of Rs.908.36", "LICN", 825.78, 908.36),
    ("Price Adjusted – NHDL",
     "Price of Nepal Hydro Developers Ltd. (NHDL) is Rs.687.04 for 8% Bonus Shares on previous closing price of Rs.742",
     "NHDL", 687.04, 742.0),
    ("Price Adjusted - HATHY",
     "Adjusted price of HATHY is 724.09 for 10% bonus shares on previous closing price of Rs. 796.50",
     "HATHY", 724.09, 796.50),
    ("Price Adjusted - Chilime Hydropower Company Limited (CHCL)",
     "Adjusted price of Chilime Hydropower Company Limited is Rs.519.82 for 10% bonus shares on previous closing "
     "price of Rs.571.80", "CHCL", 519.82, 571.80),
    ("Price Adjusted",
     "Adjusted price of NMB50 is Rs.10.46 for 15% cash dividend on previous closing price of Rs.11.96",
     "NMB50", 10.46, 11.96),
    ("Price Adjusted - Mahalaxmi", "Adjusted price of Mahalaxmi Bikas Bank Ltd. (MLBL ) is Rs.1,377.67 for 3% Bonus "
     "shares on previous closing price of Rs.1,419.00", "MLBL", 1377.67, 1419.0),
]


@pytest.mark.parametrize("title, body, sym, adj, prev", CASES)
def test_parse_price_adjustment_variants(title, body, sym, adj, prev):
    r = parse_price_adjustment(title, body)
    assert r["symbol"] == sym
    assert r["adjusted_price"] == pytest.approx(adj) and r["prev_close"] == pytest.approx(prev)
    assert r["factor"] == pytest.approx(adj / prev)


def test_non_adjustment_and_garbage_are_ignored():
    body = "Adjusted price of X is Rs.1 for y on previous closing price of Rs.2"
    assert parse_price_adjustment("AGM notice - NABIL", body) is None
    assert parse_price_adjustment("Price Adjusted - ABC", "no numbers here") is None
