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
        'task_type': 'gaze_estimation',
        'dataset_category': datasets.DATASET_CATEGORY_REGRESSION_GAZE,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_REGRESSION_GAZE]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_REGRESSION_GAZE]['input_dataset'],
        'postprocess': postproc_transforms.get_transform_regression_gaze()
    }

    quant_params_proto_path_disable_option = {constants.ADVANCED_OPTIONS_QUANT_FILE_KEY: ''}

    pipeline_configs = {
        #################################################################
        # ONNX models/models/gaze_model/regnet/0829_gaze.onnx
        #################################################################
        'pha2-gaze-0001':utils.dict_update(
            common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(
                resize=96, crop=96, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='./pha2_models/gaze/pha2_gaze_vgg16_0001.onnx'),
            model_info=dict(metric_reference={'mse': 0.01}, model_shortlist=None)  
        ),
        
        'pha2-gaze-0002':utils.dict_update(
            common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(
                resize=96, crop=96, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='./pha2_models/gaze/pha2_gaze_regnet_x_800mf_0002.onnx'),
            model_info=dict(metric_reference={'mse': 0.01}, model_shortlist=None)  
        ),
        
        'pha2-gaze-0003_default_mean_scale':utils.dict_update(
            common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(
                resize=96, crop=96, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0, Y_extract=True),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(128.0,), 
                    input_scale=(0.0078125,)
                    ),
                
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='./pha2_models/gaze/2025_09_24_Y_channel.onnx'),
            model_info=dict(metric_reference={'mse': 0.01}, model_shortlist=None)  
        ),
        'pha2-gaze-0003_Y':utils.dict_update(
            common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(
                resize=96, crop=96, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0, Y_extract=True),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(116.95,), # 116.95 
                    input_scale=(0.0131,) # 0.0131
                    ),

                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='./pha2_models/gaze/2025_09_24_Y_channel.onnx'),
            model_info=dict(metric_reference={'mse': 0.01}, model_shortlist=None)  
        ),
        'gaze-0004_Y':utils.dict_update(
            common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(
                resize=96, crop=96, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2_Y_extract',
                resize_with_pad=False, add_flip_image=False, pad_color=0, Y_extract=False),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(119.065664,), # mean=[119.065664], std=[64.324008]
                    input_scale= (64.324008,)  #(0.015542,) # 
                    ),

                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='/home/edgeai/code/edgeai-benchmark/3d_model/gaze/best_model_dV7_E14.5.onnx'),
            model_info=dict(metric_reference={'mse': 0.01}, model_shortlist=None)  
        ),
           
        'gaze-0004_Y_norm_with_255':utils.dict_update(
            common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(
                resize=96, crop=96, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2_Y_extract',
                resize_with_pad=False, add_flip_image=False, pad_color=0, Y_extract=False),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(119.065664*255,), # mean=[119.065664], std=[64.324008]
                    input_scale= (1/(64.324008*255),)  #(0.015542,) # 
                    ),

                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='/home/edgeai/code/edgeai-benchmark/3d_model/gaze/1104_v4.onnx'),
            model_info=dict(metric_reference={'mse': 0.01}, model_shortlist=None)  
        ),
        
        '0005-gaze':utils.dict_update(
            common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(
                resize=96, crop=96, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2_Y_extract',
                resize_with_pad=False, add_flip_image=False, pad_color=0, Y_extract=False),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(119.065664*255,), # mean=[119.065664], std=[64.324008]
                    input_scale= (1/(64.324008*255),)  #(0.015542,) # 
                    ),

                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='/home/edgeai/code/edgeai-benchmark/3d_model/gaze/2025_12_22.onnx'),
            model_info=dict(metric_reference={'mse': 0.01}, model_shortlist=None)  
        ),

    }

    return pipeline_configs
