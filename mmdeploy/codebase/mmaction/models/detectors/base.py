# Copyright (c) OpenMMLab. All rights reserved.

#from typing import List, Optional, Tuple, Union

from typing import Union, List

from mmaction.utils import OptSampleList
from mmdeploy.core import FUNCTION_REWRITER, mark
from mmengine.structures import InstanceData

import torch



@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.BaseDetector.forward')
def base_detector__forward(self,
                           inputs: List[torch.Tensor],
                           data_samples: OptSampleList = None,
                           mode: str = 'tensor',
                           **kwargs):
    """Rewrite `forward` of Recognizer2D for default backend.

    Args:
        inputs (List[torch.Tensor]): A list of input tensors input tensor with shape
            (N, C, ...) in general.
        data_samples (List[``ActionDataSample``], optional): The
            annotation data of every samples. Defaults to None.
        mode (str): Return what kind of value. Defaults to ``tensor``.

    Returns:
        return a list of `ActionDataSample`
    """

    """
    assert mode == 'predict'

    feats, predict_kwargs = self.extract_feat(inputs, test_mode=True)
    cls_scores = self.cls_head(feats, **predict_kwargs)
    num_segs = cls_scores.shape[0] // len(data_samples)
    cls_scores = self.cls_head.average_clip(cls_scores, num_segs=num_segs)

    return cls_scores
    """
    images = inputs[0]
    proposals = inputs[1]

    # Backbone
    x = self.extract_feat(images)

    # Head
    rpn_results_list = [
        InstanceData(bboxes=tensor, scores=torch.Tensor([[1.0]]))
        for tensor in proposals
    ]
    
    results_list = self.roi_head.predict(
        x, rpn_results_list, data_samples)

    # Convert to tensors
    bboxes = torch.stack([result.bboxes for result in results_list])
    scores = torch.stack([result.scores for result in results_list])

    return bboxes, scores
