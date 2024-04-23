# Copyright (c) OpenMMLab. All rights reserved.

#from typing import List, Optional, Tuple, Union

from mmaction.utils import OptSampleList
from torch import Tensor
#import torch.nn.functional as F
from torch import nn

from mmdeploy.core import FUNCTION_REWRITER, mark

from mmdet.structures.bbox import bbox2roi
import numpy as np


#from mmdeploy.codebase.mmdet.models.detectors import two_stage

import inspect
def GRUNER():
    frame = inspect.currentframe().f_back
    file_name = frame.f_code.co_filename
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno
    print(f"===== GRUNER ===== : {file_name}:{function_name}:{line_number}")

GRUNER()


@FUNCTION_REWRITER.register_rewriter(
    'mmaction.models.roi_heads.bbox_heads.BBoxHeadAVA.forward')
def bbox_head__forward(self, x: Tensor) -> Tensor:    
    """Computes the classification logits given ROI features."""
    GRUNER()

    if self.dropout_before_pool and self.dropout_ratio > 0:
        x = self.dropout(x)

    #self.aap = nn.AdaptiveAvgPool3d((1,16,22))
    print("GRUNER ,<<<<<<<<", x.shape)
    self.aap = nn.AvgPool3d(kernel_size=(1, 1, 1), stride=(1, 1, 1))
    x = self.aap(x)#F.adaptive_avg_pool3d(x, (1, 16, 22))
    print("GRUNER ,<<<<<<<<", x.shape)
    #self.amp = nn.AdaptiveMaxPool3d((1,1,1))
    self.amp = nn.AvgPool3d(kernel_size=(1, 16, 22), stride=(1, 16, 22))
    x = self.amp(x)#F.adaptive_max_pool3d(x, (1, 1, 1))
    print("GRUNER ,<<<<<<<<", x.shape)

    if not self.dropout_before_pool and self.dropout_ratio > 0:
        x = self.dropout(x)

    x = x.view(x.size(0), -1)
    cls_score = self.fc_cls(x)
    return cls_score
