from edgeai_benchmark import datasets, preprocess, postprocess, metrics, constants,  utils, sessions
import onnxruntime as ort
import cv2

def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)
    tflite_session_type = settings.get_session_type(constants.MODEL_TYPE_TFLITE)
    mxnet_session_type = settings.get_session_type(constants.MODEL_TYPE_MXNET)

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

    # configs for each model pipeline
    common_cfg = {
        'task_type': 'face_landmark_24',
        'dataset_category': datasets.DATASET_CATEGORY_REGRESSION_FACELM,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_REGRESSION_FACELM]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_REGRESSION_FACELM]['input_dataset'],
        'postprocess': postproc_transforms.get_transform_regression_facelm()
    }

    quant_params_proto_path_disable_option = {constants.ADVANCED_OPTIONS_QUANT_FILE_KEY: ''}

    pipeline_configs = {
        #################################################################
        # ONNX models/models/gaze_model/regnet/0829_gaze.onnx
        #################################################################
        'pha-2-facelm-24-viz-0001':utils.dict_update(
             common_cfg,
             preprocess=preproc_transforms.get_transform_onnx(
                 resize=192, crop=192, data_layout=constants.NCHW, 
                 reverse_channels=False, backend='cv2',
                 resize_with_pad=False, add_flip_image=False, pad_color=0),
             session=onnx_session_type(
                 **sessions.get_onnx_session_cfg(
                     settings, 
                     work_dir=work_dir),
                 runtime_options=settings.runtime_options_onnx_p2(),
                 model_path='./pha2_models/face_landmark_24keypoints_viz/pha-2-facelm-24-viz-0001.onnx'),
             
             model_info=dict(metric_reference={'mse': 0.01}, model_shortlist=None)  
         ),
        
        'pha-2-facelm-24-viz-0002':utils.dict_update(
             common_cfg,
             preprocess=preproc_transforms.get_transform_onnx(
                 resize=192, crop=192, data_layout=constants.NCHW, 
                 reverse_channels=False, backend='cv2',
                 resize_with_pad=False, add_flip_image=False, pad_color=0),
             session=onnx_session_type(
                 **sessions.get_onnx_session_cfg(
                     settings, 
                     work_dir=work_dir),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                    {
                        # 'advanced_options:output_feature_16bit_names_list':'/stem/stem.0/Conv, /trunk_output/block4/block4-4/f/c/c.0/Conv'
                        }),
                 model_path='./pha2_models/face_landmark_24keypoints_viz/pha-2-facelm-24-viz-0002.onnx'),
            #  metric=metrics.MetricMSE(),  # regression 평가를 위한 MSE metric 추가
             model_info=dict(metric_reference={'mse': 0.01}, model_shortlist=None)  
         ),
        'pha-2-facelm-24-viz-0003':utils.dict_update(
             common_cfg,
             preprocess=preproc_transforms.get_transform_onnx(
                 resize=192, crop=192, data_layout=constants.NCHW, 
                 reverse_channels=False, backend='cv2',
                 resize_with_pad=False, add_flip_image=False, pad_color=0),
             session=onnx_session_type(
                 **sessions.get_onnx_session_cfg(
                     settings, 
                     work_dir=work_dir),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                    {
                        # 'advanced_options:output_feature_16bit_names_list':'/stem/stem.0/Conv, /trunk_output/block4/block4-4/f/c/c.0/Conv'
                        }),
                 model_path='./pha2_models/face_landmark_24keypoints_viz/pha-2-facelm-24-viz-0003.onnx'),
            #  metric=metrics.MetricMSE(),  # regression 평가를 위한 MSE metric 추가
             model_info=dict(metric_reference={'mse': 0.01}, model_shortlist=None)  # MSE reference 값으로
         ),
        'pha-2-facelm-24-viz-0004':utils.dict_update(
             common_cfg,
             preprocess=preproc_transforms.get_transform_onnx(
                 resize=192, crop=192, data_layout=constants.NCHW, 
                 reverse_channels=False, backend='cv2',
                 resize_with_pad=False, add_flip_image=False, pad_color=0),
             session=onnx_session_type(
                 **sessions.get_onnx_session_cfg(
                     settings, 
                     work_dir=work_dir),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                    {
                        # 'advanced_options:output_feature_16bit_names_list':'/stem/stem.0/Conv, /trunk_output/block4/block4-4/f/c/c.0/Conv'
                        
                        }
                    ),
                 model_path='./pha2_models/face_landmark_24keypoints_viz/pha-2-facelm-24-viz-0004.onnx'),
            #  metric=metrics.MetricMSE(),  # regression 평가를 위한 MSE metric 추가
             model_info=dict(metric_reference={'mse': 0.01}, model_shortlist=None)  # MSE reference 값으로
         ),
        'pha-2-facelm-24-viz-0005':utils.dict_update(
             common_cfg,
             preprocess=preproc_transforms.get_transform_onnx(
                 resize=192, crop=192, data_layout=constants.NCHW, 
                 reverse_channels=False, backend='cv2',
                 resize_with_pad=False, add_flip_image=False, pad_color=0),
             session=onnx_session_type(
                 **sessions.get_onnx_session_cfg(
                     settings, 
                     work_dir=work_dir),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                    {
                        # 'advanced_options:output_feature_16bit_names_list':'/stem/stem.0/Conv, /trunk_output/block4/block4-4/f/c/c.0/Conv'
                     }
                    ),
                 model_path='./pha2_models/face_landmark_24keypoints_viz/pha-2-facelm-24-viz-0005.onnx'),
            #  metric=metrics.MetricMSE(),  # regression 평가를 위한 MSE metric 추가
             model_info=dict(metric_reference={'mse': 0.01}, model_shortlist=None)  # MSE reference 값으로
         ),
        'pha-2-facelm-24-viz-0006':utils.dict_update(
             common_cfg,
             preprocess=preproc_transforms.get_transform_onnx(
                 resize=192, crop=192, data_layout=constants.NCHW, 
                 reverse_channels=False, backend='cv2',
                 resize_with_pad=False, add_flip_image=False, pad_color=0),
             session=onnx_session_type(
                 **sessions.get_onnx_session_cfg(
                     settings, 
                     work_dir=work_dir),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                    {
                        # 'advanced_options:output_feature_16bit_names_list':'/stem/stem.0/Conv, /trunk_output/block4/block4-4/f/c/c.0/Conv'
                     }
                    ),
                 model_path='./pha2_models/face_landmark_24keypoints_viz/pha-2-facelm-24-viz-0006.onnx'),
            #  metric=metrics.MetricMSE(),  # regression 평가를 위한 MSE metric 추가
             model_info=dict(metric_reference={'mse': 0.01}, model_shortlist=None)  # MSE reference 값으로
         ),
    }

    return pipeline_configs
