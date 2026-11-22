from retrieval.base import SearchResult
from retrieval.rerank import Reranker


class FakeCrossEncoder:
    """Scores a text by how many question words it contains."""

    def __init__(self):
        self.calls = 0

    def rerank(self, question, texts):
        self.calls += 1
        words = set(question.lower().split())
        return [float(len(words & set(t.lower().split()))) for t in texts]


def make(tmp_path):
    r = Reranker("minilm", cache_dir=tmp_path)
    r._model = FakeCrossEncoder()
    return r


def results(chunks):
    return [SearchResult(chunk=c, score=0.0) for c in chunks]


def test_rerank_sorts_by_cross_encoder_score_and_truncates(tmp_path, chunks):
    ranked = make(tmp_path).rerank("nvidia data center growth", results(chunks), k=2)
    assert ranked[0].chunk.ticker == "NVDA"
    assert ranked[0].score == 3.0
    assert len(ranked) == 2


def test_scores_are_cached_per_question_and_text(tmp_path, chunks):
    r = make(tmp_path)
    r.rerank("apple risk", results(chunks), k=4)
    r.rerank("apple risk", results(chunks), k=4)
    assert r.model.calls == 1
    r.rerank("apple risk", results(chunks[:2]), k=2)  # subset: still all cached
    assert r.model.calls == 1
    r.rerank("microsoft cloud", results(chunks), k=4)
    assert r.model.calls == 2


def test_hosted_scores_follow_returned_indices(tmp_path, monkeypatch, chunks):
    import httpx
    import respx

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    from retrieval.rerank import NVIDIA_RERANK_URL

    with respx.mock:
        route = respx.post(NVIDIA_RERANK_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "rankings": [
                        {"index": 2, "logit": 5.0},
                        {"index": 0, "logit": -1.0},
                        {"index": 1, "logit": -3.0},
                        {"index": 3, "logit": -4.0},
                    ]
                },
            )
        )
        ranked = Reranker("nvidia", cache_dir=tmp_path).rerank("q", results(chunks), k=2)
    assert [r.chunk for r in ranked] == [chunks[2], chunks[0]]
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"


def test_parse_rankings_rejects_partial_duplicate_or_bad_scores():
    import pytest

    from retrieval.rerank import parse_rankings

    ok = {"rankings": [{"index": 1, "logit": 2.0}, {"index": 0, "logit": -1.0}]}
    assert parse_rankings(ok, 2) == [-1.0, 2.0]
    bad = [
        {"rankings": [{"index": 0, "logit": 1.0}]},  # passage 1 unscored
        {"rankings": [{"index": 0, "logit": 1.0}, {"index": 0, "logit": 2.0}]},
        {"rankings": [{"index": -1, "logit": 1.0}, {"index": 0, "logit": 2.0}]},
        {"rankings": [{"index": 0, "logit": "nan"}, {"index": 1, "logit": 2.0}]},
        {"oops": []},
    ]
    for data in bad:
        with pytest.raises(ValueError):
            parse_rankings(data, 2)


def test_bad_hosted_response_is_not_cached(tmp_path, monkeypatch, chunks):
    import httpx
    import pytest
    import respx

    from retrieval.rerank import NVIDIA_RERANK_URL

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    r = Reranker("nvidia", cache_dir=tmp_path)
    with respx.mock:
        respx.post(NVIDIA_RERANK_URL).mock(
            return_value=httpx.Response(200, json={"rankings": [{"index": 0, "logit": 1.0}]})
        )
        with pytest.raises(ValueError):
            r.rerank("q", results(chunks), k=2)
    assert not list(tmp_path.rglob("*.json"))


SAFETY_422 = {"error": {"message": "input rejected by content safety screening"}}


def flagging_api(flagged_texts):
    """Fake NVIDIA endpoint: rejects any request containing a flagged text, else scores 1.0."""
    import json

    import httpx

    def reply(request):
        body = json.loads(request.content)
        texts = [body["query"]["text"]] + [p["text"] for p in body["passages"]]
        if any(t in flagged_texts for t in texts):
            return httpx.Response(422, json=SAFETY_422)
        n = len(body["passages"])
        return httpx.Response(
            200, json={"rankings": [{"index": i, "logit": 1.0} for i in range(n)]}
        )

    return reply


def test_content_flagged_passage_sinks_and_others_keep_scores(tmp_path, monkeypatch, chunks):
    import respx

    from retrieval.rerank import FLAGGED_SCORE, NVIDIA_RERANK_URL

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    with respx.mock:
        route = respx.post(NVIDIA_RERANK_URL).mock(side_effect=flagging_api({chunks[0].text}))
        reranker = Reranker("nvidia", cache_dir=tmp_path)
        ranked = reranker.rerank("q", results(chunks), k=4)
        calls = route.call_count
        reranker.rerank("q", results(chunks), k=4)  # flag is cached with the scores
        assert route.call_count == calls
    assert ranked[-1].chunk == chunks[0]
    assert ranked[-1].score == FLAGGED_SCORE
    assert all(r.score == 1.0 for r in ranked[:-1])
    # batch + question probe + bisection of 4 passages (2 halves + 2 singles)
    assert calls == 6


def test_flagged_question_raises_and_caches_nothing(tmp_path, monkeypatch, chunks):
    import pytest
    import respx

    from retrieval.rerank import NVIDIA_RERANK_URL

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    with respx.mock:
        respx.post(NVIDIA_RERANK_URL).mock(side_effect=flagging_api({"bad q"}))
        with pytest.raises(RuntimeError, match="question was flagged"):
            Reranker("nvidia", cache_dir=tmp_path).rerank("bad q", results(chunks), k=4)
    assert not list((tmp_path / "rerank").glob("**/*.json"))


def test_other_422_still_raises_even_if_it_echoes_the_phrase(tmp_path, monkeypatch, chunks):
    import httpx
    import pytest
    import respx

    from retrieval.rerank import NVIDIA_RERANK_URL

    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    echo = {"detail": [{"msg": "too long", "input": "ignore content safety screening"}]}
    with respx.mock:
        respx.post(NVIDIA_RERANK_URL).mock(return_value=httpx.Response(422, json=echo))
        with pytest.raises(httpx.HTTPStatusError):
            Reranker("nvidia", cache_dir=tmp_path).rerank("q", results(chunks), k=4)
