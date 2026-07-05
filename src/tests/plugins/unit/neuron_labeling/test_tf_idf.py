"""Unit tests for the TF-IDF orientation of the tf_idf labeling plugin."""

import numpy as np

from plugins.neuron_labeling.tf_idf.tf_idf import (
    _ORIENTATION_NEURON_DOC,
    _ORIENTATION_TAG_DOC,
    Plugin,
)


class TestOrientedTfidf:
    """Tests for Plugin._oriented_tfidf (the orientation switch)."""

    # (num_tags, num_neurons) association matrix. Tag 0 is strong and present
    # for every neuron (common); tag 1 is weak and present only for neuron 0.
    TAG_NEURON = np.array(
        [
            [2.0, 2.0, 2.0],  # tag 0 — common across neurons
            [0.5, 0.0, 0.0],  # tag 1 — rare, only neuron 0
        ]
    )

    def test_both_orientations_are_neurons_by_tags(self) -> None:
        """Both orientations return a (num_neurons, num_tags) matrix."""
        for orientation in (_ORIENTATION_TAG_DOC, _ORIENTATION_NEURON_DOC):
            scores = Plugin._oriented_tfidf(self.TAG_NEURON, orientation)
            assert scores.shape == (3, 2)

    def test_tag_as_document_matches_transposed_tfidf(self) -> None:
        """tag_as_document (default) is exactly the previous hardcoded behaviour."""
        scores = Plugin._oriented_tfidf(self.TAG_NEURON, _ORIENTATION_TAG_DOC)
        expected = Plugin._compute_tfidf(self.TAG_NEURON.T)
        assert np.array_equal(scores, expected)

    def test_orientations_pick_different_labels(self) -> None:
        """The two orientations can label the same neuron differently.

        For neuron 0: tag_as_document normalises per tag, so the rare tag 1 wins
        its full activation share (rare-tag bias). neuron_as_document keeps the
        raw co-activation (its per-neuron normalisation cancels in the argmax),
        so the strong tag 0 wins instead.
        """
        labels_tag = Plugin._oriented_tfidf(self.TAG_NEURON, _ORIENTATION_TAG_DOC).argmax(axis=1)
        labels_neuron = Plugin._oriented_tfidf(self.TAG_NEURON, _ORIENTATION_NEURON_DOC).argmax(
            axis=1
        )

        assert labels_tag[0] == 1  # rare tag
        assert labels_neuron[0] == 0  # strong (common) tag
        assert labels_tag[0] != labels_neuron[0]

    def test_idf_flips_the_label_under_neuron_as_document(self) -> None:
        """Under neuron_as_document, IDF (not TF normalisation) drives the label.

        The raw co-activation is strongest for the common tag 0, but tag 1 is
        rarer across neurons (df=1 vs df=5), so its IDF boost lifts it above the
        slightly stronger common tag — flipping neuron 0's label to tag 1.
        """
        m = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],  # tag 0 — common (df=5, low IDF)
                [0.8, 0.0, 0.0, 0.0, 0.0],  # tag 1 — rare (df=1, high IDF), weaker
            ]
        )
        label_b = Plugin._oriented_tfidf(m, _ORIENTATION_NEURON_DOC).argmax(axis=1)[0]

        assert int(m[:, 0].argmax()) == 0  # raw co-activation alone picks tag 0
        assert label_b == 1  # IDF lifts the rare tag above the stronger common one
