"""Unit tests for app.core.plugin_discovery.naming.

These cover the pure, standalone formatters that do not require mocking.
``make_plugin_display_name`` and ``format_step_run_name`` are exercised
elsewhere (registry / engine tests); this module focuses on
``format_pipeline_run_name`` and ``make_dataset_label``.
"""

import datetime

from app.core.plugin_discovery.naming import (
    format_pipeline_run_name,
    make_dataset_label,
)

_FIXED_NOW = datetime.datetime(2026, 5, 31, 14, 7)


class TestFormatPipelineRunName:
    """Tests for format_pipeline_run_name (pure, clock injected)."""

    def test_without_name_uses_plain_format(self) -> None:
        """Verify an empty name yields the no-name timestamped form."""
        result = format_pipeline_run_name("", _FIXED_NOW)
        assert result == "Pipeline Run [ 31/05/2026 | 14:07 ]"

    def test_with_name_inserts_label(self) -> None:
        """Verify a name is woven in after a ' | ' separator."""
        result = format_pipeline_run_name("Baseline ELSA", _FIXED_NOW)
        assert result == "Pipeline Run | Baseline ELSA [ 31/05/2026 | 14:07 ]"

    def test_whitespace_only_name_treated_as_empty(self) -> None:
        """Verify a whitespace-only name falls back to the no-name form."""
        result = format_pipeline_run_name("   ", _FIXED_NOW)
        assert result == "Pipeline Run [ 31/05/2026 | 14:07 ]"

    def test_name_is_stripped(self) -> None:
        """Verify surrounding whitespace is trimmed from the name."""
        result = format_pipeline_run_name("  My Run  ", _FIXED_NOW)
        assert result == "Pipeline Run | My Run [ 31/05/2026 | 14:07 ]"

    def test_timestamp_renders_zero_padded(self) -> None:
        """Verify single-digit day/month/hour/minute are zero-padded."""
        now = datetime.datetime(2026, 1, 2, 3, 4)
        result = format_pipeline_run_name("", now)
        assert result == "Pipeline Run [ 02/01/2026 | 03:04 ]"


class TestMakeDatasetLabel:
    """Tests for make_dataset_label (pure, no mocking)."""

    def test_strips_loader_suffix_and_keeps_casing(self) -> None:
        """Verify the impl segment is taken and ``_loader`` is stripped."""
        result = make_dataset_label("dataset_loading.movieLens_loader.movieLens_loader")
        assert result == "movieLens"

    def test_handles_other_loader(self) -> None:
        """Verify casing/digits are preserved for another loader."""
        result = make_dataset_label("dataset_loading.lastFm1k_loader.lastFm1k_loader")
        assert result == "lastFm1k"

    def test_without_loader_suffix_returns_impl_segment(self) -> None:
        """Verify a missing ``_loader`` suffix leaves the name unchanged."""
        result = make_dataset_label("dataset_loading.steamGames.steamGames")
        assert result == "steamGames"
