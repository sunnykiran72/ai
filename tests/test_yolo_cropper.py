import unittest

import numpy as np
from PIL import Image

from ai.modules.wardrobe.yolo_cropper import YoloCropper


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeMasks:
    def __init__(self, data):
        self.data = _FakeTensor(data)


class _FakeBoxes:
    def __init__(self, data):
        self.data = _FakeTensor(data)


class _FakeResult:
    def __init__(self):
        self.masks = _FakeMasks(
            [
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                ]
            ]
        )
        # x1, y1, x2, y2, conf, class
        self.boxes = _FakeBoxes([[0, 0, 1, 1, 0.95, 0]])
        self.names = {0: "top"}


class TestYoloCropper(unittest.TestCase):
    def test_detect_and_crop(self):
        cropper = YoloCropper(predictor=lambda _rgb, _conf, _iou: [_FakeResult()])
        image = Image.new("RGB", (2, 2), (200, 100, 50))

        instances = cropper.detect_instances(image)
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["label"], "top")

        crops = cropper.get_crops(image, instances)
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0]["image"].size, (2, 2))


if __name__ == "__main__":
    unittest.main()

