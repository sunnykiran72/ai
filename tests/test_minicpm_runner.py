import os
import sys
import unittest

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.minicpm_runner import MiniCPMVRunner


class _StubRunner(MiniCPMVRunner):
    def __init__(self):
        self.garment_max_new_tokens = 640
        self.garment_min_words = 200
        self.calls = []

    def _run_prompt(self, image, instruction, max_new_tokens):
        self.calls.append(instruction)
        if len(self.calls) == 1:
            return "short garment sentence"
        return " ".join(["detailed"] * 220)


class TestMiniCPMVRunnerRetry(unittest.TestCase):
    def test_describe_garment_retries_when_output_is_too_short(self):
        runner = _StubRunner()
        image = Image.new("RGB", (32, 32), "white")

        result = runner.describe_garment(image)

        self.assertGreaterEqual(len(runner.calls), 2)
        self.assertIn("previous answer was too short", runner.calls[1].lower())
        self.assertGreaterEqual(len(result.split()), 200)

    def test_describe_garment_uses_category_specific_prompt(self):
        runner = _StubRunner()
        image = Image.new("RGB", (32, 32), "white")

        runner.describe_garment(image, garment_type="outer")

        self.assertTrue(runner.calls)
        self.assertIn("outerwear-only guidance", runner.calls[0].lower())


if __name__ == "__main__":
    unittest.main()
