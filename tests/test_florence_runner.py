import unittest

from ai.core.florence_runner import FlorenceRunner


class TestFlorenceRunnerTypeParsing(unittest.TestCase):
    def test_infer_dress(self):
        label, conf, reason = FlorenceRunner._infer_type_from_text("A red floral maxi dress with short sleeves.")
        self.assertEqual(label, "dress")
        self.assertGreater(conf, 0.6)
        self.assertIn("keyword_match", reason)

    def test_infer_bottom(self):
        label, conf, _ = FlorenceRunner._infer_type_from_text("Blue denim jeans with straight fit.")
        self.assertEqual(label, "bottom")
        self.assertGreater(conf, 0.6)

    def test_infer_outer(self):
        label, conf, _ = FlorenceRunner._infer_type_from_text("Black oversized jacket with zipper.")
        self.assertEqual(label, "outer")
        self.assertGreater(conf, 0.6)

    def test_infer_top_fallback(self):
        label, conf, reason = FlorenceRunner._infer_type_from_text("")
        self.assertEqual(label, "top")
        self.assertLessEqual(conf, 0.5)
        self.assertEqual(reason, "empty_caption_fallback")


if __name__ == "__main__":
    unittest.main()
