"""Unit tests for app.core.param_choices."""

import pytest

from app.core.param_choices import resolve_dependent_choices


class TestResolveDependentChoices:
    """Tests for resolve_dependent_choices."""

    def test_returns_choices_for_known_value(self) -> None:
        """Verify a known provider resolves to its models as choices."""
        choices = resolve_dependent_choices("embedder_models", "openai")

        assert choices
        assert all(set(c) == {"label", "value"} for c in choices)
        assert all(c["label"] == c["value"] for c in choices)

    def test_empty_value_returns_empty(self) -> None:
        """Verify a falsy controlling value yields no options."""
        assert resolve_dependent_choices("embedder_models", None) == []
        assert resolve_dependent_choices("embedder_models", "") == []

    def test_unknown_value_returns_empty(self) -> None:
        """Verify an unrecognised value is swallowed into an empty list."""
        assert resolve_dependent_choices("embedder_models", "no_such_provider") == []

    def test_unknown_resolver_key_raises(self) -> None:
        """Verify an unregistered resolver key raises KeyError."""
        with pytest.raises(KeyError):
            resolve_dependent_choices("no_such_resolver", "openai")
