#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging
import os
import sys
from collections.abc import Iterable
from typing import Set, Union

import huggingface_hub
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import ResizeShortestEdge
from detrex.data import DetrDatasetMapper
from detrex.modeling import ema
from PIL import Image

_log = logging.getLogger(__name__)


class CustomDetRexDatasetMapper(DetrDatasetMapper):
    """Dataset mapper from Detrex - which does not read the image."""

    def __call__(self, dataset_dict):
        """
        Parameters
        ----------    
        dataset_dict: metadata of one image
        
        Returns
        -------
        a dict in a format that builtin detectron2 accept
        """
        assert "image" in dataset_dict
        image = dataset_dict["image"]

        image, transforms = T.apply_transform_gens(self.augmentation, image)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        dataset_dict["transforms"] = transforms

        return dataset_dict


class DetRexLayoutPredictor:
    """
    Document layout prediction using safe tensors
    """

    def __init__(
            self,
            artifact_path: str = "bebouky/deterex-custom-model",
            device: str = "cpu",
            num_threads: int = 4,
            base_threshold: float = 0.3,
            blacklist_classes: Set[str] = set(),
    ):
        """
        Provide the artifact path that contains the LayoutModel file

        Parameters
        ----------
        artifact_path: Path for the model torch file.
        device: (Optional) device to run the inference.
        num_threads: (Optional) Number of threads to run the inference if device = 'cpu'

        Raises
        ------
        FileNotFoundError when the model's torch file is missing
        """

        # Initialize classes map:
        self._classes_map = {
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10: "Title",
            11: "Document Index",
            12: "Code",
            13: "Checkbox-Selected",
            14: "Checkbox-Unselected",
            15: "Form",
            16: "Key-Value Region",
        }

        # Blacklisted classes
        self._black_classes = blacklist_classes  # set(["Form", "Key-Value Region"])

        # Set basic params
        self._threshold = base_threshold  # Score threshold
        self._image_size = 640
        self._size = np.asarray([[self._image_size, self._image_size]],
                                dtype=np.int64)

        # Set number of threads for CPU
        self._device = torch.device(device)
        self._num_threads = num_threads
        if device == "cpu":
            torch.set_num_threads(self._num_threads)

        # Model file and configurations
        self._st_fn = huggingface_hub.snapshot_download(artifact_path)

        sys.path.insert(0, self._st_fn)

        # Load config and instantiate model
        cfg = LazyConfig.load(
            os.path.join(
                self._st_fn,
                "projects/dino_dinov2/configs/COCO/dino_dinov2_b_12ep.py"))

        self.model = instantiate(cfg.model)
        self.model.to(device)
        self.model.training = False

        # image augmentation at test time - follows config
        augmentation = [
            ResizeShortestEdge(
                short_edge_length=800,
                max_size=1333,
            )
        ]

        # Instantiate the mapper
        self.data_mapper = CustomDetRexDatasetMapper(
            is_train=False,
            augmentation=augmentation,
            augmentation_with_crop=None,
            img_format="RGB",  # match your config
            mask_on=False,
        )

        # load previous checkpoint
        DetectionCheckpointer(
            self.model, **ema.may_get_ema_checkpointer(cfg, self.model)).load(
                os.path.join(self._st_fn, "model_final.pth"))

        _log.debug("DetRex LayoutPredictor settings: {}".format(self.info()))

    def info(self) -> dict:
        """
        Get information about the configuration of LayoutPredictor
        """
        info = {
            "safe_tensors_file": self._st_fn,
            "device": self._device.type,
            "num_threads": self._num_threads,
            "image_size": self._image_size,
            "threshold": self._threshold,
        }
        return info

    @torch.inference_mode()
    def predict(self, orig_img: Union[Image.Image,
                                      np.ndarray]) -> Iterable[dict]:
        """
        Predict bounding boxes for a given image.
        The origin (0, 0) is the top-left corner and the predicted bbox coords are provided as:
        [left, top, right, bottom]

        Parameter
        ---------
        origin_img: Image to be predicted as a PIL Image object or numpy array.

        Yield
        -----
        Bounding box as a dict with the keys: "label", "confidence", "l", "t", "r", "b"

        Raises
        ------
        TypeError when the input image is not supported
        """
        # Convert image format
        if isinstance(orig_img, Image.Image):
            page_img = orig_img.convert("RGB")
        elif isinstance(orig_img, np.ndarray):
            page_img = Image.fromarray(orig_img).convert("RGB")
        else:
            raise TypeError("Not supported input image format")

        w, h = page_img.size

        batched_page_img = {
            "image": utils.convert_PIL_to_numpy(page_img, "RGB"),
            "height": h,
            "width": w,
        }
        mapped_img = self.data_mapper(batched_page_img)  # (batched_page_img)

        results = self.model([mapped_img])
        result = results[0]['instances']

        for score, label_id, box in zip(result.scores, result.pred_classes,
                                        result.pred_boxes):
            score = float(score.item())

            if score < self._threshold:
                continue

            label_id = int(label_id.item())  # + 1  # Advance the label_id
            label_str = self._classes_map[label_id]

            # Filter out blacklisted classes
            if label_str in self._black_classes:
                continue

            bbox_float = [float(b.item()) for b in box]

            l = min(w, max(0, bbox_float[0]))
            t = min(h, max(0, bbox_float[1]))
            r = min(w, max(0, bbox_float[2]))
            b = min(h, max(0, bbox_float[3]))

            yield {
                "l": l,
                "t": t,
                "r": r,
                "b": b,
                "label": label_str,
                "confidence": score,
            }
