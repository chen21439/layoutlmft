import torch
from typing import List, Tuple

# ---- Detectron2 shims (only used when detectron2 is unavailable) ----
try:
    from detectron2.structures import Boxes, Instances  # type: ignore
except Exception:
    class Boxes:
        def __init__(self, tensor: torch.Tensor):
            # expect shape [N,4] in xyxy
            self.tensor = tensor.float()

        def to(self, device):
            self.tensor = self.tensor.to(device)
            return self

        def __len__(self):
            return self.tensor.shape[0]

        def __repr__(self):
            return f"Boxes(shape={tuple(self.tensor.shape)}, device={self.tensor.device})"

    class Instances:
        def __init__(self, image_size: Tuple[int, int]):
            self.image_size = image_size
            self._fields = {}

        def set(self, name, value):
            self._fields[name] = value

        def get(self, name, default=None):
            return self._fields.get(name, default)

        def __setattr__(self, name, value):
            if name in {"image_size", "_fields"}:
                super().__setattr__(name, value)
            else:
                self._fields[name] = value

        def __getattr__(self, name):
            if name in {"image_size", "_fields"}:
                return super().__getattribute__(name)
            if name in self._fields:
                return self._fields[name]
            raise AttributeError(name)

        def to(self, device):
            for k, v in list(self._fields.items()):
                if hasattr(v, "to"):
                    self._fields[k] = v.to(device)
                elif isinstance(v, (list, tuple)):
                    self._fields[k] = type(v)(
                        t.to(device) if hasattr(t, "to") else t for t in v
                    )
            return self
# --------------------------------------------------------------------

import torch

from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeTransform, TransformList


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]


def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]


def load_image(image_path):
    image = read_image(image_path, format="BGR")
    h = image.shape[0]
    w = image.shape[1]
    img_trans = TransformList([ResizeTransform(h=h, w=w, new_h=224, new_w=224)])
    image = torch.tensor(img_trans.apply_image(image).copy()).permute(2, 0, 1)  # copy to make it writeable
    return image, (w, h)

