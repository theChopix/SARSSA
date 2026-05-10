"""Unit tests for plugins.steering._steer.compute_steered_recommendations."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix


def _make_csr(rows: list[list[int]], num_items: int) -> csr_matrix:
    """Build a csr_matrix from per-row lists of interacted item indices.

    Args:
        rows: Outer list is users; each inner list is the column
            indices of items they interacted with.
        num_items: Total number of items (column dimension).

    Returns:
        csr_matrix: ``(len(rows), num_items)`` binary interaction
            matrix.
    """
    indptr = [0]
    indices: list[int] = []
    for row in rows:
        indices.extend(row)
        indptr.append(len(indices))
    data = [1] * len(indices)
    return csr_matrix((data, indices, indptr), shape=(len(rows), num_items))


def _make_recommender(top_indices: list[int]) -> MagicMock:
    """Build a mock model whose ``recommend`` returns deterministic indices.

    Args:
        top_indices: Item column indices the recommender should
            return as its top-k.

    Returns:
        MagicMock: A mock with ``to``, ``eval``, and ``recommend``
            methods configured.  ``recommend`` returns
            ``(scores, indices)`` matching the real signature.
    """
    model = MagicMock()
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    scores = torch.tensor([[0.0] * len(top_indices)])
    indices = torch.tensor([top_indices])
    model.recommend = MagicMock(return_value=(scores, indices))
    return model


class TestComputeSteeredRecommendations:
    """Tests for the shared steering helper."""

    @patch("plugins.steering._steer.SteeredModel")
    def test_returns_three_artifact_lists(
        self,
        mock_steered_cls: MagicMock,
    ) -> None:
        """Verify all three item-id lists are populated as expected."""
        from plugins.steering._steer import compute_steered_recommendations

        full_csr = _make_csr([[0, 2]], num_items=4)
        items = np.array(["item_a", "item_b", "item_c", "item_d"])
        users = np.array(["u_orig"])
        labels = {"7": "concept_a"}

        base_model = _make_recommender([3, 1])
        sae = MagicMock()
        steered = _make_recommender([1, 0])
        mock_steered_cls.return_value = steered

        result = compute_steered_recommendations(
            full_csr=full_csr,
            items=items,
            users=users,
            base_model=base_model,
            sae=sae,
            neuron_labels=labels,
            user_id=0,
            neuron_id="7",
            alpha=0.4,
            k=2,
            device="cpu",
        )

        assert result["interacted_items"] == ["item_a", "item_c"]
        assert result["original_recommendations"] == ["item_d", "item_b"]
        assert result["steered_recommendations"] == ["item_b", "item_a"]
        assert result["neuron_id"] == 7
        assert result["label"] == "concept_a"
        assert result["user_id"] == 0
        assert result["user_original_id"] == "u_orig"

    @patch("plugins.steering._steer.SteeredModel")
    def test_steered_model_built_with_alpha_and_neuron(
        self,
        mock_steered_cls: MagicMock,
    ) -> None:
        """Verify the steered model is constructed with the right knobs."""
        from plugins.steering._steer import compute_steered_recommendations

        full_csr = _make_csr([[0]], num_items=2)
        items = np.array(["a", "b"])
        users = np.array(["u"])

        base_model = _make_recommender([1])
        sae = MagicMock()
        steered = _make_recommender([0])
        mock_steered_cls.return_value = steered

        compute_steered_recommendations(
            full_csr=full_csr,
            items=items,
            users=users,
            base_model=base_model,
            sae=sae,
            neuron_labels={"3": "x"},
            user_id=0,
            neuron_id="3",
            alpha=0.7,
            k=1,
            device="cpu",
        )

        mock_steered_cls.assert_called_once_with(base_model, sae, alpha=0.7)
        call_kwargs = steered.recommend.call_args.kwargs
        assert call_kwargs["neuron_ids"] == [3]
        assert call_kwargs["k"] == 1
        assert call_kwargs["mask_interactions"] is True

    @patch("plugins.steering._steer.SteeredModel")
    def test_user_id_out_of_range_raises(
        self,
        _mock_steered_cls: MagicMock,
    ) -> None:
        """Verify out-of-range user_id is rejected before any model call."""
        from plugins.steering._steer import compute_steered_recommendations

        full_csr = _make_csr([[0]], num_items=2)
        kwargs: dict[str, Any] = {
            "full_csr": full_csr,
            "items": np.array(["a", "b"]),
            "users": np.array(["u"]),
            "base_model": MagicMock(),
            "sae": MagicMock(),
            "neuron_labels": {"0": "x"},
            "user_id": 5,
            "neuron_id": "0",
            "alpha": 0.1,
            "k": 1,
            "device": "cpu",
        }
        with pytest.raises(ValueError, match="user_id"):
            compute_steered_recommendations(**kwargs)

    @patch("plugins.steering._steer.SteeredModel")
    def test_unknown_neuron_id_raises(
        self,
        _mock_steered_cls: MagicMock,
    ) -> None:
        """Verify a neuron id not in the labels mapping raises."""
        from plugins.steering._steer import compute_steered_recommendations

        full_csr = _make_csr([[0]], num_items=2)
        kwargs: dict[str, Any] = {
            "full_csr": full_csr,
            "items": np.array(["a", "b"]),
            "users": np.array(["u"]),
            "base_model": MagicMock(),
            "sae": MagicMock(),
            "neuron_labels": {"0": "x"},
            "user_id": 0,
            "neuron_id": "99",
            "alpha": 0.1,
            "k": 1,
            "device": "cpu",
        }
        with pytest.raises(ValueError, match="99"):
            compute_steered_recommendations(**kwargs)

    @patch("plugins.steering._steer.SteeredModel")
    def test_alpha_out_of_range_raises(
        self,
        _mock_steered_cls: MagicMock,
    ) -> None:
        """Verify alpha outside [0, 1] is rejected."""
        from plugins.steering._steer import compute_steered_recommendations

        full_csr = _make_csr([[0]], num_items=2)
        kwargs: dict[str, Any] = {
            "full_csr": full_csr,
            "items": np.array(["a", "b"]),
            "users": np.array(["u"]),
            "base_model": MagicMock(),
            "sae": MagicMock(),
            "neuron_labels": {"0": "x"},
            "user_id": 0,
            "neuron_id": "0",
            "alpha": 1.5,
            "k": 1,
            "device": "cpu",
        }
        with pytest.raises(ValueError, match="alpha"):
            compute_steered_recommendations(**kwargs)

    @patch("plugins.steering._steer.SteeredModel")
    def test_user_original_id_is_string(
        self,
        mock_steered_cls: MagicMock,
    ) -> None:
        """Verify a non-string user identifier is coerced to str for output."""
        from plugins.steering._steer import compute_steered_recommendations

        full_csr = _make_csr([[]], num_items=1)
        items = np.array(["a"])
        users = np.array([42])
        labels = {"0": "x"}

        base_model = _make_recommender([0])
        sae = MagicMock()
        mock_steered_cls.return_value = _make_recommender([0])

        result = compute_steered_recommendations(
            full_csr=full_csr,
            items=items,
            users=users,
            base_model=base_model,
            sae=sae,
            neuron_labels=labels,
            user_id=0,
            neuron_id="0",
            alpha=0.0,
            k=1,
            device="cpu",
        )

        assert isinstance(result["user_original_id"], str)
        assert result["user_original_id"] == "42"
