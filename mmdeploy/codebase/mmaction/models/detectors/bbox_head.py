# Copyright (c) OpenMMLab. All rights reserved.

#from typing import List, Optional, Tuple, Union

from mmaction.utils import OptSampleList
from torch import Tensor
import torch.nn.functional as F

from mmdeploy.core import FUNCTION_REWRITER, mark

from mmdet.structures.bbox import bbox2roi


@FUNCTION_REWRITER.register_rewriter(
    'mmaction.models.roi_heads.bbox_heads.BBoxHeadAVA.forward')
def bbox_head__forward(self, x: Tensor) -> Tensor:    
    """Computes the classification logits given ROI features."""

    if self.dropout_before_pool and self.dropout_ratio > 0:
        x = self.dropout(x)

    x = F.avg_pool3d(x, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    x = F.max_pool3d(x, kernel_size=(1, 16, 22), stride=(1, 16, 22))

    if not self.dropout_before_pool and self.dropout_ratio > 0:
        x = self.dropout(x)

    x = x.view(x.size(0), -1)
    cls_score = self.fc_cls(x)
    return cls_score
