import unittest

import numpy as np

from ai.main import _nearest_color_label, _split_outfit_signature_from_parsing


class SplitOutfitDetectionTests(unittest.TestCase):
    def test_detects_exposed_waist_two_piece_set(self):
        parsing = np.zeros((220, 140), dtype=np.uint8)
        parsing[24:92, 28:112] = 4   # upper
        parsing[112:212, 24:116] = 5  # skirt

        signature = _split_outfit_signature_from_parsing(parsing)

        self.assertTrue(bool(signature.get("detected")))
        self.assertEqual(signature.get("reason"), "split_gap_detected")
        self.assertGreater(int(signature.get("gap_px", 0)), 8)

    def test_does_not_flag_one_piece_dress_as_split_set(self):
        parsing = np.zeros((220, 140), dtype=np.uint8)
        parsing[24:212, 24:116] = 7  # dress

        signature = _split_outfit_signature_from_parsing(parsing)

        self.assertFalse(bool(signature.get("detected")))
        self.assertIn(signature.get("reason"), {"missing_top_or_bottom", "bridge_present_or_gap_small"})

    def test_low_chroma_green_gray_resolves_to_green_family_not_gray(self):
        # Approximate muted sage/gray-green from the failing dress example.
        self.assertIn(_nearest_color_label((133, 141, 130)), {"green", "olive"})

    def test_dark_cool_low_chroma_resolves_to_navy_or_plum_not_black(self):
        # Approximate inky navy-black blazer tones from the top color failure case.
        self.assertIn(_nearest_color_label((18, 15, 27)), {"navy", "plum"})


if __name__ == "__main__":
    unittest.main()
