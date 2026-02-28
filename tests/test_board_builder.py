import unittest

from PIL import Image

from ai.modules.vto.board_builder import BoardBuilder


class TestBoardBuilder(unittest.TestCase):
    def test_build_board_shape(self):
        builder = BoardBuilder(canvas_size=(512, 512))
        imgs = [
            Image.new("RGB", (100, 200), (255, 0, 0)),
            Image.new("RGB", (100, 200), (0, 255, 0)),
            Image.new("RGB", (100, 200), (0, 0, 255)),
        ]
        board = builder.build_board(imgs)
        self.assertEqual(board.size, (512, 512))

    def test_layout_count(self):
        builder = BoardBuilder(canvas_size=(512, 512))
        rects = builder.get_layout_rects(4)
        self.assertEqual(len(rects), 4)


if __name__ == "__main__":
    unittest.main()

