_base_ = ['./video-detection_static.py', '../../_base_/backends/tensorrt.py']

onnx_config = dict(
    input_shape=None,
    input_names=['input', 'proposals'],
    output_names=['bboxes', 'scores']
)

num_proposals=10

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 32, 256, 340],
                    opt_shape=[1, 3, 32, 256, 340],
                    max_shape=[1, 3, 32, 256, 340]),
                proposals=dict(
                    min_shape=[1, num_proposals, 4],
                    opt_shape=[1, num_proposals, 4],
                    max_shape=[1, num_proposals, 4]))
        )
    ])
