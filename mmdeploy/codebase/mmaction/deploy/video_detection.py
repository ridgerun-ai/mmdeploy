# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from operator import itemgetter
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import torch
from mmengine.dataset import pseudo_collate
from mmengine.model import BaseDataPreprocessor

from mmdeploy.codebase.base import BaseTask
from mmdeploy.utils import Task, get_root_logger
from mmdeploy.utils.config_utils import get_input_shape
from .mmaction import MMACTION_TASK


@MMACTION_TASK.register_module(Task.VIDEO_DETECTION.value)
class VideoDetection(BaseTask):
    """VideoDetection task class.

    Args:
        model_cfg (Config): Original PyTorch model config file.
        deploy_cfg (Config): Deployment config file or loaded Config
            object.
        device (str): A string represents device type.
    """

    def __init__(self, model_cfg: mmengine.Config, deploy_cfg: mmengine.Config,
                 device: str):
        super(VideoDetection, self).__init__(model_cfg, deploy_cfg, device)

    def build_backend_model(self,
                            model_files: Sequence[str] = None,
                            **kwargs) -> torch.nn.Module:
        """Initialize backend model.

        Args:
            model_files (Sequence[str]): Input model files.

        Returns:
            nn.Module: An initialized backend model.
        """
        from .video_recognition_model import build_video_recognition_model
        data_preprocessor = self.model_cfg.model.data_preprocessor
        data_preprocessor.setdefault('type', 'mmaction.ActionDataPreprocessor')
        model = build_video_recognition_model(
            model_files,
            self.model_cfg,
            self.deploy_cfg,
            device=self.device,
            data_preprocessor=data_preprocessor)
        model.to(self.device)
        return model.eval()

