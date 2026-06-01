"""Unit tests for app.core.plugin_discovery.naming.

These cover the pure, standalone formatters that do not require mocking.
``make_plugin_display_name`` and ``format_step_run_name`` are exercised
elsewhere (registry / engine tests); this module focuses on
``format_pipeline_run_name``.
"""

import datetime

from app.core.plugin_discovery.naming import format_pipeline_run_name

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
