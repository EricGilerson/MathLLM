"""Tests for data generation pipeline."""

import re
import tempfile
from pathlib import Path

from mathllm.config import DataConfig
from mathllm.data.generator import ArithmeticDataGenerator
from mathllm.data.negative_examples import NegativeExampleSampler
from mathllm.data.templates import TEMPLATES


class TestTemplates:
    def test_all_ops_have_templates(self):
        for op in ["add", "sub", "mul", "exp", "div"]:
            assert op in TEMPLATES
            assert len(TEMPLATES[op]) >= 20, f"{op} has too few templates"

    def test_addition_templates_have_50_plus(self):
        assert len(TEMPLATES["add"]) >= 50

    def test_template_placeholders(self):
        """All templates should be renderable with {a}, {b}, {result} placeholders."""
        for op in ["add", "sub", "mul", "exp", "div"]:
            for i, template in enumerate(TEMPLATES[op]):
                # Should not raise KeyError — all placeholders must be {a}, {b}, or {result}
                try:
                    rendered = template.format(a=1, b=2, result=3)
                except KeyError as e:
                    raise AssertionError(
                        f"Template {op}[{i}] has unknown placeholder {e}: {template}"
                    )


class TestNegativeExamples:
    def test_generates_examples(self):
        sampler = NegativeExampleSampler(seed=42)
        examples = sampler.sample(100)
        assert len(examples) == 100

    def test_examples_contain_numbers(self):
        sampler = NegativeExampleSampler(seed=42)
        examples = sampler.sample(50)
        for ex in examples:
            assert re.search(r"\d", ex), f"No number in: {ex}"

    def test_no_arithmetic(self):
        """Negative examples should not contain arithmetic operators between numbers."""
        sampler = NegativeExampleSampler(seed=42)
        examples = sampler.sample(50)
        # Basic check: none should match "number op number = number"
        arith_pattern = re.compile(r"\d+\s*[+\-*/]\s*\d+\s*=\s*\d+")
        for ex in examples:
            assert not arith_pattern.search(ex), f"Arithmetic found in negative: {ex}"


class TestArithmeticDataGenerator:
    def _make_config(self, **kwargs):
        defaults = dict(
            num_positive=100,
            num_negative=100,
            num_edge_cases=20,
            max_digits=5,
            max_value=1_000_000_000,
            seed=42,
            output_dir="",
        )
        defaults.update(kwargs)
        return DataConfig(**defaults)

    def test_generates_correct_count(self):
        config = self._make_config()
        gen = ArithmeticDataGenerator(config)
        examples = gen.generate_dataset()
        # Should be approximately num_positive + num_negative + num_edge_cases
        expected = config.num_positive + config.num_negative + config.num_edge_cases
        assert len(examples) >= expected * 0.8  # Allow some failed generations
        assert len(examples) <= expected * 1.1

    def test_addition_correctness(self):
        """Spot-check that simple addition examples have correct answers."""
        config = self._make_config(num_positive=500, num_negative=0, num_edge_cases=0)
        gen = ArithmeticDataGenerator(config)
        examples = gen.generate_dataset()

        # Match only clean "a + b = result" (not part of multi-step)
        pattern = re.compile(r"^(\d+)\s*\+\s*(\d+)\s*=\s*(-?\d+)$")
        found = 0
        for ex in examples:
            match = pattern.match(ex.strip())
            if match:
                a, b, result = int(match.group(1)), int(match.group(2)), int(match.group(3))
                assert a + b == result, f"Wrong addition: {a} + {b} != {result}"
                found += 1

        assert found > 0, "No addition examples found matching pattern"

    def test_multiplication_in_range(self):
        """All multiplication results should be within representable range."""
        config = self._make_config(num_positive=500, num_negative=0, num_edge_cases=0)
        gen = ArithmeticDataGenerator(config)
        examples = gen.generate_dataset()

        pattern = re.compile(r"(\d+)\s*[*×x]\s*(\d+)\s*=\s*(\d+)")
        for ex in examples:
            match = pattern.search(ex)
            if match:
                result = int(match.group(3))
                assert result <= config.max_value, f"Result {result} exceeds max"

    def test_save_and_load(self):
        config = self._make_config(num_positive=10, num_negative=10, num_edge_cases=5)
        gen = ArithmeticDataGenerator(config)
        examples = gen.generate_dataset()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = gen.save_dataset(examples, tmpdir)
            assert path.exists()
            # Check it's valid JSONL
            import json
            with open(path) as f:
                loaded = [json.loads(line) for line in f]
            assert len(loaded) == len(examples)
            assert all("text" in item for item in loaded)

    def test_reproducible(self):
        """Same seed should produce same data."""
        config = self._make_config(seed=123)
        gen1 = ArithmeticDataGenerator(config)
        gen2 = ArithmeticDataGenerator(config)
        ex1 = gen1.generate_dataset()
        ex2 = gen2.generate_dataset()
        assert ex1 == ex2
