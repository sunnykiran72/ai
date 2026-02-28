import unittest

import numpy as np
from PIL import Image

from ai.modules.wardrobe.human_parser import HumanParser


class TestHumanParser(unittest.TestCase):
    def test_parse_requires_parser_fn(self):
        parser = HumanParser()
        with self.assertRaises(RuntimeError):
            parser.parse(Image.new("RGB", (2, 2), (0, 0, 0)))

    def test_category_masks(self):
        parsing = np.array(
            [
                [5, 9, 6],
                [7, 12, 10],
            ],
            dtype=np.int32,
        )
        parser = HumanParser(parser_fn=lambda _img: parsing)
        masks = parser.build_category_masks(parsing)
        self.assertTrue(masks["top"][0, 0])
        self.assertTrue(masks["bottom"][0, 1])
        self.assertTrue(masks["dress"][0, 2])
        self.assertTrue(masks["outer"][1, 0])


if __name__ == "__main__":
    unittest.main()

