# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from edgeai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics
import cv2

def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

    # configs for each model pipeline
    # TIDL has post processing (simlar to object detection post processing) inside it for keypoint estimation
    # These models use that keypoint post processing
    # YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss
    # Debapriya Maji, Soyeb Nagori, Manu Mathew, Deepak Poddar
    # https://arxiv.org/abs/2204.06806
    common_cfg = {
        'task_type': 'keypoint_detection',
        'dataset_category': datasets.DATASET_CATEGORY_COCOKPTS,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_COCOKPTS]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_COCOKPTS]['input_dataset'],
        'postprocess': postproc_transforms.get_transform_human_pose_estimation_onnx() 
    }

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        ################# onnx models ###############################
        # yolox based keypoint/pose estimation - post processing is handled completely by TIDL
        'kd-7060':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=settings.runtime_options_onnx_p2(
                        det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                         'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/keypoint/coco/edgeai-yolox/yolox_s_pose_ti_lite_640_20220301_model.prototxt',
                        'advanced_options:output_feature_16bit_names_list': '/0/backbone/backbone/stem/stem.0/act/Relu_output_0, /0/head/cls_preds.0/Conv_output_0, /0/head/reg_preds.0/Conv_output_0, /0/head/obj_preds.0/Conv_output_0, /0/head/kpts_preds.0/Conv_output_0, /0/head/cls_preds.1/Conv_output_0, /0/head/reg_preds.1/Conv_output_0, /0/head/obj_preds.1/Conv_output_0, /0/head/kpts_preds.1/Conv_output_0, /0/head/cls_preds.2/Conv_output_0, /0/head/reg_preds.2/Conv_output_0, /0/head/obj_preds.2/Conv_output_0, /0/head/kpts_preds.2/Conv_output_0'},
                        fast_calibration=True), 
                model_path=f'{settings.models_path}/vision/keypoint/coco/edgeai-yolox/yolox_s_pose_ti_lite_640_20220301_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            metric=dict(label_offset_pred=1), #TODO: add this for other models as well?
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':49.6, 'accuracy_ap50%':78.0}, model_shortlist=10)
        ),
        'pha-kpts-0001':utils.dict_update({
            'task_type': 'keypoint_detection',
            'dataset_category': datasets.DATASET_CATEGORY_PHAKPTS,
            'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['calibration_dataset'],
            'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['input_dataset'],                
            'postprocess': postproc_transforms.get_transform_human_pose_estimation_onnx() 
        },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=(480, 640), 
                crop=(480, 640), 
                reverse_channels=True,
                data_layout=constants.NCHW,
                backend='cv2',
                interpolation=cv2.INTER_LINEAR,
                resize_with_pad=[True, "corner"],
                add_flip_image=False, pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(
                                    settings, 
                                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': 'models/bodyKPs/PHA_BodyKeyPointDetection_dev_Edgeai_Yolovx_V1.0/v3_v4_combined_finetune_on_coco_model.prototxt'},
                    # 'advanced_options:output_feature_16bit_names_list': '1033, 711, 712, 713, 727, 728, 728, 743, 744, 745'},
                    fast_calibration=True),
                model_path=f'models/bodyKPs/PHA_BodyKeyPointDetection_dev_Edgeai_Yolovx_V1.0/v3_v4_combined_finetune_on_coco_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            metric=dict(label_offset_pred=1),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':9.6, 'accuracy_ap50%':78.0}, model_shortlist=10)),
        'pha-kpts-0002':utils.dict_update({
            'task_type': 'keypoint_detection',
            'dataset_category': datasets.DATASET_CATEGORY_PHAKPTS,
            'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['calibration_dataset'],
            'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['input_dataset'],                
            'postprocess': postproc_transforms.get_transform_human_pose_estimation_onnx() 
        },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=(480, 640), 
                crop=(480, 640), 
                reverse_channels=True,
                data_layout=constants.NCHW,
                backend='cv2',
                interpolation=cv2.INTER_LINEAR,
                resize_with_pad=[True, "corner"],
                add_flip_image=False, pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(
                                    settings, 
                                    work_dir=work_dir,
                                    input_optimization=True),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': 'models/bodyKPs/PHA_BodyKeyPointDetection_dev_Edgeai_Yolovx_V1.0/v3_v4_combined_finetune_on_coco_model.prototxt'},
                    # 'advanced_options:output_feature_16bit_names_list': '1033, 711, 712, 713, 727, 728, 728, 743, 744, 745'},
                    fast_calibration=True),
                model_path=f'models/bodyKPs/with_nms.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            metric=dict(label_offset_pred=1),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':9.6, 'accuracy_ap50%':78.0}, model_shortlist=10)),
        'pha-kpts-0003':utils.dict_update({
                'task_type': 'keypoint_detection',
                'dataset_category': datasets.DATASET_CATEGORY_PHAKPTS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['input_dataset'],                
                'postprocess': postproc_transforms.get_transform_human_pose_estimation_onnx() 
            },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=(480, 640), 
                crop=(480, 640), 
                reverse_channels=True,
                data_layout=constants.NCHW,
                backend='cv2',
                interpolation=cv2.INTER_LINEAR,
                resize_with_pad=[True, "corner"],
                add_flip_image=False, pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(
                                    settings, 
                                    work_dir=work_dir,
                                    input_optimization=True),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': 'models/bodyKPs/0106_V2/Body_keypoint_2025_01_06.prototxt'},
                    # 'advanced_options:output_feature_16bit_names_list': '1033, 711, 712, 713, 727, 728, 728, 743, 744, 745'},
                    fast_calibration=True),
                model_path=f'models/bodyKPs/0106_V2/Body_keypoint_2025_01_06.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            metric=dict(label_offset_pred=1),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':9.6, 'accuracy_ap50%':78.0}, model_shortlist=10)
        ),
        'pha-kpts-0004':utils.dict_update({
                'task_type': 'keypoint_detection',
                'dataset_category': datasets.DATASET_CATEGORY_PHAKPTS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['input_dataset'],                
                'postprocess': postproc_transforms.get_transform_human_pose_estimation_onnx() 
            },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=(480, 640), 
                crop=(480, 640), 
                reverse_channels=True,
                data_layout=constants.NCHW,
                backend='cv2',
                interpolation=cv2.INTER_LINEAR,
                resize_with_pad=[True, "corner"],
                add_flip_image=False, pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(
                                    settings, 
                                    work_dir=work_dir,
                                    input_optimization=True),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': 'models/bodyKPs/0108_V3/Body_keypoint_2025_01_08.prototxt'},
                    # 'advanced_options:output_feature_16bit_names_list': '1033, 711, 712, 713, 727, 728, 728, 743, 744, 745'},
                    fast_calibration=True),
                model_path=f'models/bodyKPs/0108_V3/Body_keypoint_2025_01_08.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            metric=dict(label_offset_pred=1),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':9.6, 'accuracy_ap50%':78.0}, model_shortlist=10)
        ),
        'pha-kpts-0005':utils.dict_update({
                'task_type': 'keypoint_detection',
                'dataset_category': datasets.DATASET_CATEGORY_PHAKPTS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['input_dataset'],                
                'postprocess': postproc_transforms.get_transform_human_pose_estimation_onnx() 
            },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=(480, 640), 
                crop=(480, 640), 
                reverse_channels=True,
                data_layout=constants.NCHW,
                backend='cv2',
                interpolation=cv2.INTER_LINEAR,
                resize_with_pad=[True, "corner"],
                add_flip_image=False, pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(
                                    settings, 
                                    work_dir=work_dir,
                                    input_optimization=True),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': 'models/bodyKPs/0122_V5/Body_keypoint_2025_01_22.prototxt'},
                    # 'advanced_options:output_feature_16bit_names_list': '1033, 711, 712, 713, 727, 728, 728, 743, 744, 745'},
                    fast_calibration=True),
                model_path=f'models/bodyKPs/0122_V5/Body_keypoint_2025_01_22.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            metric=dict(label_offset_pred=1),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':9.6, 'accuracy_ap50%':78.0}, model_shortlist=10)
        ),
        
        'pha-kpts-0004_8bit':utils.dict_update({
                'task_type': 'keypoint_detection',
                'dataset_category': datasets.DATASET_CATEGORY_PHAKPTS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['input_dataset'],                
                'postprocess': postproc_transforms.get_transform_human_pose_estimation_onnx() 
            },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=(480, 640), 
                crop=(480, 640), 
                reverse_channels=True,
                data_layout=constants.NCHW,
                backend='cv2',
                interpolation=cv2.INTER_LINEAR,
                resize_with_pad=[True, "corner"],
                add_flip_image=False, pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(
                                    settings, 
                                    work_dir=work_dir,
                                    input_optimization=True),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': 'models/bodyKPs/0108_V3/Body_keypoint_2025_01_08.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '/0/backbone/backbone/stem/stem.0/act/Relu_output_0, /0/head/cls_preds.0/Conv_output_0, /0/head/reg_preds.0/Conv_output_0, /0/head/obj_preds.0/Conv_output_0, /0/head/kpts_preds.0/Conv_output_0, /0/head/cls_preds.1/Conv_output_0, /0/head/reg_preds.1/Conv_output_0, /0/head/obj_preds.1/Conv_output_0, /0/head/kpts_preds.1/Conv_output_0, /0/head/cls_preds.2/Conv_output_0, /0/head/reg_preds.2/Conv_output_0, /0/head/obj_preds.2/Conv_output_0, /0/head/kpts_preds.2/Conv_output_0'},
                    fast_calibration=True),
                model_path=f'models/bodyKPs/0108_V3/Body_keypoint_2025_01_08.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            metric=dict(label_offset_pred=1),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':9.6, 'accuracy_ap50%':78.0}, model_shortlist=10)
        ),
        
        ###############################################pha 2 ######################################################################################################
            
            'pha2-kpts-0001':utils.dict_update({
            'task_type': 'keypoint_detection',
            'dataset_category': datasets.DATASET_CATEGORY_PHAKPTS,
            'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['calibration_dataset'],
            'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAKPTS]['input_dataset'],                
            'postprocess': postproc_transforms.get_transform_human_pose_estimation_onnx() 
        },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=(384, 640), 
                crop=(384, 640), 
                reverse_channels=True,
                data_layout=constants.NCHW,
                backend='cv2',
                interpolation=cv2.INTER_LINEAR,
                resize_with_pad=[True, "corner"],
                add_flip_image=False, pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(
                                    settings, 
                                    work_dir=work_dir,
                                    input_optimization=True),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': './pha2_models/keypoints/test/test0923.prototxt',
                    # 'advanced_options:output_feature_16bit_names_list': '/0/backbone/backbone/stem/stem.0/act/Relu_output_0, /0/head/cls_preds.0/Conv_output_0, /0/head/reg_preds.0/Conv_output_0, /0/head/obj_preds.0/Conv_output_0, /0/head/kpts_preds.0/Conv_output_0, /0/head/cls_preds.1/Conv_output_0, /0/head/reg_preds.1/Conv_output_0, /0/head/obj_preds.1/Conv_output_0, /0/head/kpts_preds.1/Conv_output_0, /0/head/cls_preds.2/Conv_output_0, /0/head/reg_preds.2/Conv_output_0, /0/head/obj_preds.2/Conv_output_0, /0/head/kpts_preds.2/Conv_output_0'
                    },
                    fast_calibration=True),
                model_path=f'./pha2_models/keypoints/test/test0923.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            metric=dict(label_offset_pred=1),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':9.6, 'accuracy_ap50%':78.0}, model_shortlist=10)
        ),
                
                
    }    
    
    return pipeline_configs