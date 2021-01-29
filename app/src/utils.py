import importlib
import numpy
import os
import torch.utils.data
import torchvision.transforms.functional
import torch
import torch._six
import torch.distributed as dist
import torch.utils.data
import torchvision
import omegaconf
import albumentations
import cv2
import matplotlib.pyplot

def load_obj(obj_path: str, default_obj_path: str = ""):
    """Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`."
        )
    return getattr(module_obj, obj_name)

def get_model(cfg):
    model = load_obj(cfg.model.backbone.class_name)
    model = model(**cfg.model.backbone.params)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    head = load_obj(cfg.model.head.class_name)

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = head(in_features, cfg.model.head.params.num_classes)

    return model

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=0)

