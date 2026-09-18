from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmc_aqa import BMCAQA, MODALITIES, ModalityMemoryBank
from bmc_aqa.losses import LossConfig, completion_loss, core_losses, retrieval_loss
from bmc_aqa.models.calibration import FeatureCalibration
from bmc_aqa.protocol import aggregate_metrics, sample_masks


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(1)
        self.dims = dict(zip(MODALITIES, (4, 6, 8)))
        self.features = {name: torch.randn(3, 3, dim) for name, dim in self.dims.items()}
        self.observed = torch.eye(3)
        self.model = BMCAQA(self.dims, hidden_dim=8, heads=2, encoder_layers=1,
                           generator_layers=1, decoder_layers=1, grades=4, key_dim=4, dropout=0)
        self.memory = ModalityMemoryBank(
            ["a", "b", "c"], self.features, torch.tensor([0.2, 0.5, 0.8])
        )

    def evidence(self, features=None):
        return self.memory.retrieve(features or self.features, self.observed,
                                    self.model.encode_retrieval_key, exclude_ids=self.memory.ids)

    def test_hidden_features_and_memory_scores_do_not_affect_predictions(self):
        self.model.eval()
        corrupted = {name: value.clone() for name, value in self.features.items()}
        for index, name in enumerate(MODALITIES):
            corrupted[name][self.observed[:, index] == 0] = float("nan")
        with torch.no_grad():
            original = self.model(self.features, self.observed, self.evidence())
            self.memory.labels = torch.randn(3) * 10000
            altered = self.model(corrupted, self.observed, self.evidence(corrupted))
        for name in ("prediction", "source_prompts", "source_weights", "modality_weights"):
            self.assertTrue(torch.isfinite(altered[name]).all())
            torch.testing.assert_close(original[name], altered[name], rtol=0, atol=0)

    def test_hierarchical_weights_and_generation_equations(self):
        outputs = self.model(self.features, self.observed, self.evidence())
        valid = outputs["source_validity"]
        weights = outputs["source_weights"]
        torch.testing.assert_close(weights[~valid], torch.zeros_like(weights[~valid]))
        torch.testing.assert_close(weights[:, :, 0], self.observed)
        torch.testing.assert_close(weights.sum(-1), torch.ones(3, 3))
        torch.testing.assert_close(outputs["modality_weights"].sum(-1), torch.ones(3))
        self.assertEqual(len(self.model.hmoe.source_experts), 3)
        self.assertEqual(len(self.model.hmoe.modality_experts), 3)
        gate = outputs["generation_gate"]
        torch.testing.assert_close(outputs["generated"],
                                   gate * outputs["refined_query"] + (1 - gate) * outputs["retrieved"])
        expected_conf = torch.sigmoid(-outputs["log_variance"]) * (1 - self.observed)
        torch.testing.assert_close(outputs["source_confidence"][:, :, 1], expected_conf)
        torch.testing.assert_close(outputs["prediction"],
                                   (outputs["grade_probability"] * self.model.score.bins).sum(-1))

    def test_three_core_losses_supervise_all_new_paths(self):
        outputs = self.model(self.features, self.observed, self.evidence())
        losses = core_losses(outputs, self.memory.labels, self.observed,
                             self.model.encode_targets(self.features), self.evidence(), self.memory.labels, LossConfig())
        losses["total"].backward()
        for prefix in ("keys", "generators", "retrieval_confidence", "prompts", "fcm", "hmoe", "score"):
            gradients = [parameter.grad for name, parameter in self.model.named_parameters()
                         if name.startswith(prefix) and parameter.grad is not None]
            self.assertTrue(gradients, prefix)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients), prefix)
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0, prefix)

    def test_completion_balances_modalities_and_detaches_targets(self):
        # Modality 0 has one masked sample (E=4), modality 1 has two (E=1,9).
        generated = torch.tensor([[[[2.0]], [[1.0]], [[0.0]]],
                                  [[[99.0]], [[3.0]], [[0.0]]]], requires_grad=True)
        targets = torch.zeros_like(generated, requires_grad=True)
        observed = torch.tensor([[0., 0., 1.], [1., 0., 1.]])
        nu = torch.zeros(2, 3, requires_grad=True)
        loss = completion_loss(generated, targets, nu, observed, eta=0.2)
        torch.testing.assert_close(loss, torch.tensor(1.1 * (4 + (1 + 9) / 2) / 2))
        loss.backward()
        self.assertIsNone(targets.grad)
        self.assertEqual(float(generated.grad[1, 0]), 0)
        self.assertTrue(torch.isfinite(nu.grad).all())
        zero = completion_loss(generated, targets, nu, torch.ones_like(observed), 0.2)
        self.assertEqual(float(zero.detach()), 0)

    def test_invalid_prompts_cannot_enter_fcm_context(self):
        fcm = FeatureCalibration(3, 3, 8)
        sources = torch.randn(2, 3, 3, 4, 8)
        prompts = torch.randn(2, 3, 3, 8)
        valid = torch.zeros(2, 3, 3, dtype=torch.bool)
        valid[:, 0, 0] = True
        first, ctx, _ = fcm(sources, prompts, valid)
        changed = prompts.clone()
        changed[~valid] = 100000
        second, ctx_changed, _ = fcm(sources, changed, valid)
        torch.testing.assert_close(ctx, sources[:, 0, 0].mean(1) + prompts[:, 0, 0])
        torch.testing.assert_close(ctx, ctx_changed)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first[~valid], torch.zeros_like(first[~valid]))

    def test_missing_memory_keeps_context_generation_available(self):
        outputs = self.model(self.features, self.observed)
        self.assertFalse(outputs["source_validity"][:, :, 2].any())
        self.assertTrue(torch.isfinite(outputs["prediction"]).all())
        torch.testing.assert_close(outputs["source_weights"][:, :, 1], 1 - self.observed)

    def test_fcm_and_hmoe_ablations_are_independent(self):
        for use_fcm, use_hmoe in ((False, True), (True, False)):
            self.model.use_fcm, self.model.use_hmoe = use_fcm, use_hmoe
            outputs = self.model(self.features, self.observed, self.evidence())
            self.assertEqual(outputs["fcm_context"] is not None, use_fcm)
            self.assertTrue(torch.isfinite(outputs["prediction"]).all())

    def test_masks_are_seeded_and_always_nonempty(self):
        first = sample_masks(200, [0.3, 0.5, 0.7], torch.Generator().manual_seed(7))
        second = sample_masks(200, [0.3, 0.5, 0.7], torch.Generator().manual_seed(7))
        torch.testing.assert_close(first, second)
        self.assertTrue((first.sum(-1) > 0).all())
        for rates in ([1, 1, 1], [float("nan"), 0, 0]):
            with self.assertRaises(ValueError):
                sample_masks(2, rates)

    def test_undefined_spearman_does_not_discard_mse(self):
        result = aggregate_metrics([{"spearman": None, "mse": 10}, {"spearman": 0.5, "mse": 2}])
        self.assertEqual(result["mse"], 6)
        self.assertAlmostEqual(result["spearman"], 0.5)


class RetrievalTests(unittest.TestCase):
    def bank(self, ids=None):
        keys = torch.tensor([[1., 0.], [0.6, 0.8], [0., 1.]])
        return ModalityMemoryBank(ids or ["a", "b", "c"],
                                  {name: keys[:, None].repeat(1, 2, 1) for name in MODALITIES},
                                  torch.tensor([0.1, 0.2, 0.3]))

    def query(self):
        return {name: torch.tensor([[[1., 0.], [1., 0.]]]) for name in MODALITIES}

    def test_retrieval_statistics_and_aggregation_match_manual_values(self):
        bank = self.bank()
        result = bank.retrieve(self.query(), torch.tensor([[1., 0., 0.]]),
                               lambda name, value: F.normalize(value.mean(1), dim=-1), temperature=0.5, top_k=3)
        weights = torch.softmax(torch.tensor([2., 1.2, 0.]), 0)
        entropy = -(weights * weights.log()).sum() / math.log(3)
        torch.testing.assert_close(result.confidence_features, torch.tensor([[1., 0.4, entropy]]))
        torch.testing.assert_close(result.tokens["flow"], (bank.tokens["flow"] * weights[:, None, None]).sum(0)[None])

    def test_self_exclusion_and_exact_soft_target_cross_entropy(self):
        result = self.bank().retrieve(self.query(), torch.tensor([[1., 0., 0.]]),
                                     lambda name, value: F.normalize(value.mean(1), dim=-1), exclude_ids=["a"], top_k=3)
        self.assertEqual(result.candidate_valid.tolist(), [[False, True, True]])
        self.assertEqual(float(result.weights[0, -1]), 0)
        loss = retrieval_loss(result, torch.tensor([0.15]), torch.tensor([0.1, 0.2, 0.3]), 0.1, 0.5)
        target = torch.softmax(torch.tensor([-0.5, -1.5]), 0)
        expected = -(target * torch.log_softmax(torch.tensor([1.2, 0.]), 0)).sum()
        torch.testing.assert_close(loss, expected)

    def test_single_empty_and_duplicate_self_banks(self):
        encoder = lambda name, value: F.normalize(value.mean(1), dim=-1)
        single = ModalityMemoryBank(["only"], {name: value[:1] for name, value in self.bank().tokens.items()})
        result = single.retrieve(self.query(), torch.tensor([[1., 0., 0.]]), encoder)
        torch.testing.assert_close(result.confidence_features, torch.tensor([[1., 0., 0.]]))
        for bank in (ModalityMemoryBank([], {name: value[:0] for name, value in self.bank().tokens.items()}),
                     self.bank(["self", "self", "self"])):
            result = bank.retrieve(self.query(), torch.ones(1, 3), encoder, exclude_ids=["self"])
            self.assertFalse(result.valid_mask.any())
            torch.testing.assert_close(result.confidence_features, torch.zeros(1, 3))
            for value in result.tokens.values():
                torch.testing.assert_close(value, torch.zeros_like(value))
            self.assertEqual(float(retrieval_loss(result, torch.tensor([0.1]), torch.zeros(len(bank)), 0.1, 0.5)), 0)


if __name__ == "__main__":
    unittest.main()
