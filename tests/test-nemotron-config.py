"""Dependency-free regression for Nemotron's pre-parent architecture selection.

Evaluate the actual constructor's merge expression, without importing torch or
checkpoint conversion dependencies. Removing either fallback must fail a case.
"""
import ast
from pathlib import Path
import unittest


class NemotronConfigCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "conversion" / "nemotron.py"
        tree = ast.parse(source.read_text())
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NemotronHModel")
        ctor = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        merge = next(n.value for n in ctor.body if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == "llm_config" for t in n.targets))
        cls.expression = compile(ast.Expression(merge), str(source), "eval")

    def merged(self, hparams):
        return eval(self.expression, {"__builtins__": {}}, {"hparams": hparams})

    def test_flat_and_absent_or_null_config(self):
        for nested in ({}, {"text_config": None}, {"text_config": None, "llm_config": None}):
            with self.subTest(nested=nested):
                hp = {"num_experts_per_tok": 8, "layers_block_type": ["moe"], **nested}
                self.assertEqual(self.merged(hp)["num_experts_per_tok"], 8)
                self.assertEqual(self.merged(hp)["layers_block_type"], ["moe"])

    def test_legacy_direct_and_guessed_config(self):
        for text in ({}, {"text_config": None}, {"text_config": {}}):
            with self.subTest(text=text):
                hp = {"num_experts_per_tok": 1, "llm_config": {"num_experts_per_tok": 8}, **text}
                self.assertEqual(self.merged(hp)["num_experts_per_tok"], 8)

    def test_current_text_config_has_precedence(self):
        hp = {"num_experts_per_tok": 1, "llm_config": {"num_experts_per_tok": 4},
              "text_config": {"num_experts_per_tok": 8, "layers_block_type": ["mamba", "moe"]}}
        merged = self.merged(hp)
        self.assertEqual(merged["num_experts_per_tok"], 8)
        self.assertEqual(merged["layers_block_type"], ["mamba", "moe"])
        self.assertEqual(hp["num_experts_per_tok"], 1)


if __name__ == "__main__":
    unittest.main()
