"""Unit tests for plugins.neuron_labeling._confidence."""

import numpy as np
import scipy.sparse as sp

from plugins.neuron_labeling._confidence import (
    labels_with_confidence,
    point_biserial_matrix,
)


class TestPointBiserialMatrix:
    """Tests for the vectorised point-biserial correlation."""

    def test_shape_is_tags_by_neurons(self) -> None:
        """Output is (num_tags, num_neurons)."""
        acts = np.random.default_rng(0).random((6, 3))
        attr = sp.csr_matrix(np.random.default_rng(1).integers(0, 2, (6, 4)).astype(float))
        corr = point_biserial_matrix(acts, attr)
        assert corr.shape == (4, 3)

    def test_perfect_positive_correlation(self) -> None:
        """A neuron firing exactly on a tag's items scores +1."""
        acts = np.array([[5.0], [5.0], [0.0], [0.0]])
        attr = sp.csr_matrix(np.array([[1.0], [1.0], [0.0], [0.0]]))
        corr = point_biserial_matrix(acts, attr)
        assert np.isclose(corr[0, 0], 1.0)

    def test_negative_correlation_when_activation_avoids_tag(self) -> None:
        """A neuron firing on items lacking the tag scores negative."""
        acts = np.array([[0.0], [0.0], [5.0], [4.0]])
        attr = sp.csr_matrix(np.array([[1.0], [1.0], [0.0], [0.0]]))
        corr = point_biserial_matrix(acts, attr)
        assert corr[0, 0] < 0.0

    def test_matches_numpy_corrcoef(self) -> None:
        """Vectorised result matches a per-pair Pearson reference."""
        rng = np.random.default_rng(7)
        acts = rng.random((20, 2))
        attr_dense = rng.integers(0, 2, (20, 3)).astype(float)
        corr = point_biserial_matrix(acts, sp.csr_matrix(attr_dense))
        for t in range(3):
            for n in range(2):
                expected = np.corrcoef(attr_dense[:, t], acts[:, n])[0, 1]
                assert np.isclose(corr[t, n], expected, atol=1e-9)

    def test_zero_variance_pairs_are_zero(self) -> None:
        """Dead neurons and all/no-item tags yield 0, not NaN."""
        acts = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])  # neuron 0 dead
        attr = sp.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))  # tag 0 on all
        corr = point_biserial_matrix(acts, attr)
        assert np.all(np.isfinite(corr))
        assert corr[0, 0] == 0.0  # dead neuron
        assert corr[0, 1] == 0.0  # constant tag


class TestLabelsWithConfidence:
    """Tests for pairing labels with their assignment confidence."""

    def test_pairs_label_with_its_tag_correlation(self) -> None:
        """Confidence is the correlation of the neuron's assigned tag."""
        corr = np.array([[0.9, -0.1], [-0.2, 0.8]])  # (tags x neurons)
        labels = {0: "tagA", 1: "tagB"}
        idx = {0: 0, 1: 1}
        result, mean = labels_with_confidence(labels, idx, corr)
        assert result[0] == {"label": "tagA", "confidence": 0.9}
        assert result[1] == {"label": "tagB", "confidence": 0.8}
        assert np.isclose(mean, 0.85)

    def test_unlabelled_neuron_has_null_fields(self) -> None:
        """A neuron with no label carries None for both fields."""
        corr = np.array([[0.9, 0.0]])
        result, mean = labels_with_confidence({0: "tagA", 1: None}, {0: 0, 1: None}, corr)
        assert result[1] == {"label": None, "confidence": None}
        assert np.isclose(mean, 0.9)  # unlabelled neuron excluded from mean

    def test_mean_is_zero_when_no_labels(self) -> None:
        """No labelled neurons yields a mean confidence of 0.0."""
        corr = np.array([[0.0, 0.0]])
        result, mean = labels_with_confidence({0: None, 1: None}, {0: None, 1: None}, corr)
        assert mean == 0.0
        assert all(v == {"label": None, "confidence": None} for v in result.values())

    def test_confidence_values_are_python_floats(self) -> None:
        """Scores are plain floats so JSON serialisation stays clean."""
        corr = np.array([[np.float64(0.5)]])
        result, _ = labels_with_confidence({0: "tagA"}, {0: 0}, corr)
        assert type(result[0]["confidence"]) is float
