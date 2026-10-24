from __future__ import annotations

import httpx

from kaimei import Kaimei


def _client(seen: list) -> Kaimei:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.url.params)))
        path = request.url.path
        if path == "/v1/projects":
            summary = {"predictions_30d": 4} if request.url.params.get("include") == "summary" else None
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"object": "project", "id": "prj_1", "slug": "scorecards", "summary": summary}],
                    "total": 1,
                    "limit": 50,
                    "offset": 0,
                    "has_more": False,
                },
            )
        if path == "/v1/projects/scorecards/overview":
            return httpx.Response(
                200,
                json={
                    "object": "project.overview",
                    "project": {"object": "project", "id": "prj_1", "summary": {"predictions_30d": 4}},
                    "feature_matrix": {"features": ["income"], "rows": [], "shared": []},
                },
            )
        if path == "/v1/projects/scorecards":
            return httpx.Response(200, json={"object": "project", "id": "prj_1", "summary": None})
        return httpx.Response(404, json={"error": {"type": "not_found_error", "message": path}})

    c = Kaimei(api_key="km_test_x", base_url="https://api.kaimei.ai")
    c._transport._client = httpx.Client(
        base_url="https://api.kaimei.ai",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer km_test_x", "Accept": "application/json"},
    )
    return c


def test_projects_summary_and_overview() -> None:
    seen: list = []
    client = _client(seen)
    page = client.projects.list(include="summary")
    assert page.data[0].summary.predictions_30d == 4
    assert seen[-1] == ("/v1/projects", {"include": "summary", "limit": "50", "offset": "0"})
    client.projects.list()
    assert "include" not in seen[-1][1]
    client.projects.retrieve("scorecards", include="summary")
    assert seen[-1] == ("/v1/projects/scorecards", {"include": "summary"})
    overview = client.projects.overview("scorecards")
    assert overview.object == "project.overview"
    assert overview.feature_matrix.features == ["income"]
