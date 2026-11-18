import uuid

import pytest
from qdrant_client import models

from retrieval.qdrant_store import point_id, to_qdrant_filter


def test_point_id_is_deterministic_uuid():
    a = point_id("0000320193-24-000123:item_1a:0")
    assert a == point_id("0000320193-24-000123:item_1a:0")
    assert uuid.UUID(a).version == 5
    assert a != point_id("0000320193-24-000123:item_1a:1")


def test_no_filters_is_none():
    assert to_qdrant_filter(None) is None
    assert to_qdrant_filter({}) is None


def test_scalar_values_become_match_value():
    f = to_qdrant_filter({"ticker": "AAPL", "fiscal_year": 2024})
    assert f == models.Filter(
        must=[
            models.FieldCondition(key="ticker", match=models.MatchValue(value="AAPL")),
            models.FieldCondition(key="fiscal_year", match=models.MatchValue(value=2024)),
        ]
    )


def test_list_value_becomes_match_any():
    f = to_qdrant_filter({"ticker": ["AAPL", "MSFT"]})
    assert f == models.Filter(
        must=[models.FieldCondition(key="ticker", match=models.MatchAny(any=["AAPL", "MSFT"]))]
    )


def test_unknown_field_raises():
    with pytest.raises(ValueError, match="Unknown filter field"):
        to_qdrant_filter({"tickr": "AAPL"})


def test_text_is_not_filterable():
    with pytest.raises(ValueError):
        to_qdrant_filter({"text": "anything"})
