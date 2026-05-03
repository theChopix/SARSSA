"""Unit tests for plugins.inspection._top_k.compute_top_k_for_neuron."""

import numpy as np
import pytest
import torch

from plugins.inspection._top_k import compute_top_k_for_neuron


class TestComputeTopKForNeuron:
    """Tests for the shared top-k helper."""

    def test_returns_top_items_descending(self) -> None:
        """Verify items are sorted by activation descending."""
        items = np.array(["item_0", "item_1", "item_2", "item_3"])
        item_acts = torch.tensor(
            [
                [0.1, 0.9],
                [0.5, 0.4],
                [0.8, 0.2],
                [0.3, 0.7],
            ]
        )
        labels = {"0": "concept_a", "1": "concept_b"}

        result = compute_top_k_for_neuron(
            neuron_id="0",
            neuron_labels=labels,
            items=items,
            item_acts=item_acts,
            k=2,
        )

        assert result["top_k_item_ids"] == ["item_2", "item_1"]
        assert result["top_k_activations"] == pytest.approx([0.8, 0.5])

    def test_returns_resolved_neuron_metadata(self) -> None:
        """Verify neuron_id is parsed to int and label is included."""
        items = np.array(["x"])
        item_acts = torch.tensor([[1.0, 2.0]])
        labels = {"1": "concept_b"}

        result = compute_top_k_for_neuron(
            neuron_id="1",
            neuron_labels=labels,
            items=items,
            item_acts=item_acts,
            k=1,
        )

        assert result["neuron_id"] == 1
        assert result["label"] == "concept_b"

    def test_k_clamped_to_item_count(self) -> None:
        """Verify k larger than the number of items is clamped."""
        items = np.array(["a", "b"])
        item_acts = torch.tensor([[0.5], [0.9]])
        labels = {"0": "x"}

        result = compute_top_k_for_neuron(
            neuron_id="0",
            neuron_labels=labels,
            items=items,
            item_acts=item_acts,
            k=99,
        )

        assert result["k"] == 2
        assert len(result["top_k_item_ids"]) == 2
        assert len(result["top_k_activations"]) == 2

    def test_unknown_neuron_id_raises(self) -> None:
        """Verify a neuron id not in the mapping raises ValueError."""
        items = np.array(["a"])
        item_acts = torch.tensor([[0.0]])
        labels = {"0": "concept"}

        with pytest.raises(ValueError, match="999"):
            compute_top_k_for_neuron(
                neuron_id="999",
                neuron_labels=labels,
                items=items,
                item_acts=item_acts,
                k=1,
            )

    def test_non_positive_k_raises(self) -> None:
        """Verify k <= 0 is rejected up front."""
        items = np.array(["a"])
        item_acts = torch.tensor([[0.5]])
        labels = {"0": "x"}

        with pytest.raises(ValueError, match="positive"):
            compute_top_k_for_neuron(
                neuron_id="0",
                neuron_labels=labels,
                items=items,
                item_acts=item_acts,
                k=0,
            )

    def test_top_k_values_are_python_floats(self) -> None:
        """Verify outputs are JSON-serialisable plain Python floats."""
        items = np.array(["a", "b"])
        item_acts = torch.tensor([[0.5], [0.9]])
        labels = {"0": "x"}

        result = compute_top_k_for_neuron(
            neuron_id="0",
            neuron_labels=labels,
            items=items,
            item_acts=item_acts,
            k=2,
        )

        for activation in result["top_k_activations"]:
            assert isinstance(activation, float)
