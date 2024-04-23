# Copyright (c) OpenMMLab. All rights reserved.

#from typing import List, Optional, Tuple, Union

from mmaction.utils import OptSampleList
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER, mark

from mmdet.structures.bbox import bbox2roi
import numpy as np


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.BaseDetector.forward')
def base_recognizer__forward(self,
                             inputs: Tensor,
                             data_samples: OptSampleList = None,
                             mode: str = 'tensor',
                             **kwargs):
    return __forward_impl(self, inputs, data_samples, mode, **kwargs)


@mark('video_detector_forward', inputs='input', outputs='output')
def __forward_impl(self, inputs, data_samples, mode, **kwargs):
    """Rewrite `forward` of Recognizer2D for default backend.

    Args:
        inputs (torch.Tensor): The input tensor with shape
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

    assert self.with_bbox, 'Bbox head must be implemented.'
    x = self.extract_feat(inputs)

    assert data_samples[0].get('proposals', None) is not None
    rpn_results_list = [
        data_sample.proposals for data_sample in data_samples
    ]

    assert self.with_bbox, 'Bbox head must be implemented.'
    batch_img_metas = [
        data_sample.metainfo for data_sample in data_samples
    ]
    
    if isinstance(x, tuple):
        x_shape = x[0].shape
    else:
        x_shape = x.shape

    #assert x_shape[0] == 1, 'only accept 1 sample at test mode'
    #assert x_shape[0] == len(batch_img_metas) == len(rpn_results_list)

    proposals = [res.bboxes for res in rpn_results_list]
    rois = bbox2roi(proposals)

    bbox_feats, global_feat = self.roi_head.bbox_roi_extractor(x, rois)

    if self.roi_head.with_shared_head:
        bbox_feats = self.roi_head.shared_head(
            bbox_feats,
            feat=global_feat,
            rois=rois,
            img_metas=batch_img_metas)


    cls_score = self.roi_head.bbox_head(bbox_feats)

    return(cls_score)
