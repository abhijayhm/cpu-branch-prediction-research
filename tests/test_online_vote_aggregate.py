#!/usr/bin/env python3
"""Unit tests for online vote aggregate logic (mirrors C++ decision)."""

from __future__ import annotations

import unittest


def macc_halve(macc: int) -> int:
    """ChampSim/C++ truncates toward zero (not Python floor division)."""
    return int(macc / 2)


def vote_predict(
    bimodal_pred: bool,
    ctr_value: int,
    ctr_max: int,
    macc: int,
    model_margin: int = 8,
    vote_margin: int = 12,
    *,
    use_model_pred_on_override: bool = False,
) -> tuple[bool, bool]:
    """Return (prediction, override). Mirrors anba_online_repeat_vote.cc."""
    confident = ctr_value == 0 or ctr_value == ctr_max
    if confident:
        return bimodal_pred, False
    bim_vote = ctr_value - (ctr_max // 2)
    combined = bim_vote * 4 + macc_halve(macc)
    vote_taken = combined >= 0
    model_pred = macc >= 0
    if abs(macc) >= model_margin and abs(combined) >= vote_margin and vote_taken != bimodal_pred:
        pred = model_pred if use_model_pred_on_override else vote_taken
        return pred, True
    return bimodal_pred, False


class TestOnlineVoteAggregate(unittest.TestCase):
    def test_old_bug_model_pred_differs_from_combined(self):
        """Gate used combined direction; legacy path returned model_pred (can disagree)."""
        # ctr=3 / max=7: weak not-taken; macc=-1 -> halve=0, combined=0 -> vote taken, model not-taken
        bimodal_pred = False
        ctr_value, ctr_max, macc = 3, 7, -1
        combined = (ctr_value - ctr_max // 2) * 4 + macc_halve(macc)
        self.assertGreaterEqual(combined, 0)
        self.assertFalse(macc >= 0)
        pred_fixed, ov = vote_predict(
            bimodal_pred, ctr_value, ctr_max, macc, model_margin=0, vote_margin=0
        )
        pred_bug, _ = vote_predict(
            bimodal_pred,
            ctr_value,
            ctr_max,
            macc,
            model_margin=0,
            vote_margin=0,
            use_model_pred_on_override=True,
        )
        self.assertTrue(ov)
        self.assertTrue(pred_fixed)
        self.assertFalse(pred_bug)

    def test_no_override_when_combined_agrees_with_bimodal(self):
        pred, ov = vote_predict(True, 2, 3, 100)
        self.assertFalse(ov)
        self.assertTrue(pred)

    def test_mirror_symptom_bimodal_plus_delta_equals_other_delta(self):
        """Document: if override flips every gshare fix, acc_vote ≈ 2*acc_bim - acc_gshare."""
        acc_bim = 94.69
        acc_gshare = 99.96
        acc_vote_bad = 89.44
        expected_mirror = 2 * acc_bim - acc_gshare
        self.assertAlmostEqual(acc_vote_bad, expected_mirror, delta=0.5)


if __name__ == "__main__":
    unittest.main()
