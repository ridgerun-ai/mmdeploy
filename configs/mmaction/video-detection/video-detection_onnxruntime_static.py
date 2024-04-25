_base_ = [
    './video-detection_static.py', '../../_base_/backends/onnxruntime.py'
]

onnx_config = dict(
    input_shape=None,
    input_names=['input', 'proposals'],
    output_names=['bboxes', 'scores']
)
