"""Unit tests for utils.llm.registry.

The langchain ``ChatOpenAI`` dependency is patched so the factory
does not try to contact the OpenAI API while constructing real
chat-LLM instances.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestCreateChatLLM:
    """Tests for the ``create_chat_llm`` factory."""

    @patch("utils.llm.openai_llm.openai_llm.ChatOpenAI")
    def test_openai_returns_openai_chat_llm(
        self,
        mock_chat_openai_cls: MagicMock,
    ) -> None:
        """Verify ``"openai"`` resolves to ``OpenAIChatLLM``.

        Args:
            mock_chat_openai_cls: Patched ``ChatOpenAI`` class.
        """
        mock_chat_openai_cls.return_value = MagicMock()

        from utils.llm.openai_llm import OpenAIChatLLM
        from utils.llm.registry import create_chat_llm

        llm = create_chat_llm("openai", "gpt-4o-mini", 0.0)

        assert isinstance(llm, OpenAIChatLLM)
        assert llm.model == "gpt-4o-mini"
        assert llm.temperature == 0.0

    @patch("utils.llm.openai_llm.openai_llm.ChatOpenAI")
    def test_default_max_tokens_propagates(
        self,
        mock_chat_openai_cls: MagicMock,
    ) -> None:
        """Verify ``max_tokens`` defaults to ``5000`` when omitted.

        Args:
            mock_chat_openai_cls: Patched ``ChatOpenAI`` class.
        """
        mock_chat_openai_cls.return_value = MagicMock()

        from utils.llm.openai_llm import OpenAIChatLLM
        from utils.llm.registry import create_chat_llm

        llm = create_chat_llm("openai", "gpt-4o-mini", 0.0)

        assert isinstance(llm, OpenAIChatLLM)
        assert llm.max_tokens == 5000

    @patch("utils.llm.openai_llm.openai_llm.ChatOpenAI")
    def test_explicit_max_tokens_propagates(
        self,
        mock_chat_openai_cls: MagicMock,
    ) -> None:
        """Verify an explicit ``max_tokens`` value reaches the provider.

        Args:
            mock_chat_openai_cls: Patched ``ChatOpenAI`` class.
        """
        mock_chat_openai_cls.return_value = MagicMock()

        from utils.llm.openai_llm import OpenAIChatLLM
        from utils.llm.registry import create_chat_llm

        llm = create_chat_llm("openai", "gpt-4o-mini", 0.5, max_tokens=123)

        assert isinstance(llm, OpenAIChatLLM)
        assert llm.max_tokens == 123
        assert llm.temperature == 0.5

    def test_unknown_provider_raises_value_error(self) -> None:
        """Verify an unknown provider raises ``ValueError`` listing the known ones."""
        from utils.llm.registry import create_chat_llm

        with pytest.raises(ValueError, match="Unknown chat LLM provider 'nope'"):
            create_chat_llm("nope", "any-model", 0.0)

    def test_unknown_provider_error_lists_known_providers(self) -> None:
        """Verify the error message includes the known provider names."""
        from utils.llm.registry import create_chat_llm

        with pytest.raises(ValueError) as exc_info:
            create_chat_llm("nope", "any-model", 0.0)

        assert "openai" in str(exc_info.value)


class TestKnownProviders:
    """Tests for the ``known_providers`` helper."""

    def test_includes_openai(self) -> None:
        """Verify the registered OpenAI provider shows up in the listing."""
        from utils.llm.registry import known_providers

        providers = known_providers()

        assert "openai" in providers

    def test_returns_sorted_list(self) -> None:
        """Verify the returned list is sorted (stable order for UI use)."""
        from utils.llm.registry import known_providers

        providers = known_providers()

        assert providers == sorted(providers)
