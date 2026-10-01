"""Gemma DSpark metadata must not alter legacy DFlash sharing semantics."""
import unittest
from unittest.mock import Mock, patch

import gguf
from conversion.gemma import Gemma4DSparkModel
from conversion.qwen import DFlashModel, Qwen3Model


class GemmaDSparkMetadata(unittest.TestCase):
    def make_model(self, tied=True):
        hp = dict(attention_k_eq_v=True, layer_types=["full_attention"],
                  vocab_size=256, hidden_size=128, global_head_dim=64,
                  num_global_key_value_heads=1, tie_word_embeddings=tied,
                  final_logit_softcapping=0, target_layer_ids=[0, 2])

        def init(model, **kwargs):
            model.hparams = kwargs["hparams"]
            model.block_count = 1
            model.model_tensors = {} if tied else {"model.lm_head.weight": None}

        with patch.object(Qwen3Model, "__init__", init):
            return Gemma4DSparkModel(hparams=hp)

    def test_explicit_shared_kv_and_tied_head(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                model = self.make_model(tied)
                self.assertEqual(model.model_arch, gguf.MODEL_ARCH.DFLASH)
                model.gguf_writer = Mock()
                with patch.object(DFlashModel, "set_gguf_parameters"):
                    model.set_gguf_parameters()
                model.gguf_writer.add_bool.assert_any_call("dflash.attention.k_eq_v", True)
                model.gguf_writer.add_bool.assert_any_call("dflash.tie_word_embeddings", tied)
                model.gguf_writer.add_embedding_scale.assert_called_once_with(128 ** 0.5)
                model.gguf_writer.add_hidden_act.assert_called_once_with("gelu_pytorch_tanh")

    def test_legacy_gemma_dflash_arch_is_unchanged(self):
        with patch.object(Qwen3Model, "__init__", return_value=None):
            model = DFlashModel(hparams={"final_logit_softcapping": 30})
        self.assertEqual(model.model_arch, gguf.MODEL_ARCH.GEMMA4_DFLASH_DRAFT)

    def test_legacy_qwen_dflash_is_not_tied_implicitly(self):
        with patch.object(Qwen3Model, "__init__", return_value=None):
            model = DFlashModel(hparams={})
        self.assertEqual(model.model_arch, gguf.MODEL_ARCH.DFLASH)
        self.assertFalse(model._uses_gemma4_dspark_backbone)


if __name__ == "__main__":
    unittest.main()
