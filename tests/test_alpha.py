"""client.models.alpha: the Alpha analysis of a return forecast, and its table of dates."""

import httpx

from kaimei import Kaimei, display


def _client(handler) -> Kaimei:
    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


ALPHA = {
    "object": "alpha",
    "model_id": "mdl_1",
    "days": 120,
    "settings": {"entity": "sid", "date": "as_of", "unit": "bps", "expected_ic": 0.049},
    "summary": {"rank_ic": 0.035, "icir_annual": 7.2},
    "decay": {"status": "decaying", "severity": "high", "recent": -0.009, "z_vs_expected": -4.5},
    "daily": [
        {
            "date": "2026-11-24",
            "names": 400,
            "with_outcomes": 400,
            "rank_ic": 0.021,
            "ic": 0.03,
            "hit_rate": 0.51,
            "spread": 12.5,
            "dispersion": 10.1,
            "buckets": [-3.0, 1.0, 4.0, 2.0, 6.0],
        },
        {
            "date": "2026-11-25",
            "names": 400,
            "with_outcomes": 0,
            "rank_ic": None,
            "ic": None,
            "hit_rate": None,
            "spread": None,
            "dispersion": 9.8,
            "buckets": None,
        },
    ],
    "rolling": [{"date": "2026-11-24", "rank_ic": -0.009, "low": -0.035, "high": 0.017}],
    "horizons": {"available": True, "curve": [{"horizon": 1, "rank_ic": 0.035}], "half_life": 3.9},
    "families": [{"name": "Momentum and reversal", "status": "weakened", "z": -3.6}],
}


def test_alpha_sends_its_parameters_and_reads_as_one_row_per_date():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.url.params)))
        return httpx.Response(200, json=ALPHA)

    client = _client(handler)
    answer = client.models.alpha("us-equities-alpha-v3")
    assert seen[-1] == (
        "/v1/models/us-equities-alpha-v3/alpha",
        {"days": "120", "window": "20", "horizons": "10", "factors": "true"},
    )
    client.models.alpha(
        "toy",
        days=30,
        window=10,
        entity="ticker",
        date="as_of",
        segments=["sector", "size"],
        factors=False,
    )
    assert seen[-1][1] == {
        "days": "30",
        "window": "10",
        "horizons": "10",
        "entity": "ticker",
        "date": "as_of",
        "segments": "sector,size",
        "factors": "false",
    }
    assert answer.decay.status == "decaying" and answer.families[0].status == "weakened"
    table, columns = display.rows(answer)
    assert columns[:3] == ["date", "rank_ic", "rolling_rank_ic"]
    assert (
        table[0]["rank_ic"] == 0.021
        and table[0]["rolling_rank_ic"] == -0.009
        and "buckets" not in table[0]
    )
    # The newest date waits for its outcomes: it is listed, with nothing to correlate yet.
    assert (
        table[1]["rank_ic"] is None
        and table[1]["rolling_rank_ic"] is None
        and table[1]["names"] == 400
    )
