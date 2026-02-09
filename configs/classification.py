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
        'task_type': 'classification',
        'dataset_category': datasets.DATASET_CATEGORY_IMAGENET,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_IMAGENET]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_IMAGENET]['input_dataset'],
        'postprocess': postproc_transforms.get_transform_classification()
    }

    quant_params_proto_path_disable_option = {constants.ADVANCED_OPTIONS_QUANT_FILE_KEY: ''}


    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################jai-devkit models###############################
        # jai-devkit: classification mobilenetv1_224x224 expected_metric: 71.82% top-1 accuracy
        'cl-6060':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(
                # resize=448, crop=448, data_layout=constants.NCHW),
                resize=224, crop=224, data_layout=constants.NCHW),
                # resize=112, crop=112, data_layout=constants.NCHW),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v1_20190906.onnx'),
                # model_path="models/classification/mobileNet_v1/mobilenet_v1_20190906_resize_sym_axes_down.onnx"),
                # model_path="models/classification/mobileNet_v1/mobilenet_v1_20190906_resize_sym_scale_down.onnx"),
            model_info=dict(metric_reference={'accuracy_top1%':71.82}, model_shortlist=None)
        ),
        
        'vit-cl-0001':utils.dict_update(
            common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(
                resize=224, crop=224, data_layout=constants.NCHW),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                runtime_options=settings.runtime_options_onnx_p2(ext_options={'onnxruntime:graph_optimization_level': ort.GraphOptimizationLevel.ORT_DISABLE_ALL}),
                model_path='models/classification/vit_age_classification/deit_tiny_1.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
            )
        ),
    #     'cl-6060':utils.dict_update(
    #         {
    #     'task_type': 'classification',
    #     'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
    #     'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
    #     'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
    #     'postprocess': postproc_transforms.get_transform_classification()
    # },
        #     preprocess=preproc_transforms.get_transform_onnx(),
        #     session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_np2(),
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v1_20190906.onnx'),
        #     model_info=dict(metric_reference={'accuracy_top1%':71.82}, model_shortlist=None)
        # ),
        # jai-devkit: classification mobilenetv2_224x224 expected_metric: 72.13% top-1 accuracy
        'cl-6070':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v2_20191224.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.13}, model_shortlist=None)
        ),
        # jai-devkit: classification mobilenetv2_224x224 expected_metric: 72.13% top-1 accuracy, QAT: 71.73%
        'cl-6078':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_quant_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v2_qat-p2_20201213.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.13}, model_shortlist=None)
        ),
        # jai-devkit: classification mobilenetv2_1p4_224x224 expected_metric: 75.22% top-1 accuracy, QAT: 75.22%
        'cl-6158':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_quant_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v2_1p4_qat-p2_20210112.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.22}, model_shortlist=None)
        ),
        # jai-devkit: classification mobilenetv3_small_lite expected_metric: 62.688% top-1 accuracy
        'cl-6480':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v3_lite_small_20210429.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':62.688}, model_shortlist=30)
        ),
        # jai-devkit: classification mobilenetv3_small_lite_qat expected_metric: 61.836% top-1 accuracy
        'cl-6488':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_quant_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v3_lite_small_qat-p2_20210429.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':61.836}, model_shortlist=40)
        ),
        # jai-devkit: classification mobilenetv3_large_lite expected_metric: 72.122% top-1 accuracy
        'cl-6490':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v3_lite_large_20210507.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.122}, model_shortlist=30)
        ),
        #################torchvision models#########################
        # torchvision: classification shufflenetv2_224x224 expected_metric: 69.36% top-1 accuracy
        'cl-6080':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/shufflenet_v2_x1.0.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':69.36}, model_shortlist=90)
        ),
        # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy
        'cl-6090':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.88}, model_shortlist=20)
        ),
        # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy, QAT: 71.31%
        'cl-6098':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_quant_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_qat-p2.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.31}, model_shortlist=40)
        ),
        # torchvision: classification resnet18_224x224 expected_metric: 69.76% top-1 accuracy
        'cl-6100':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/resnet18.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':69.76}, model_shortlist=30)
        ),
        # torchvision: classification resnet50_224x224 expected_metric: 76.15% top-1 accuracy
        'cl-6110':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/resnet50.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=30)
        ),
        #################pingolh-hardnet models#########################
        'cl-6470':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/pingolh-hardnet/hardnet39ds.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.1}, model_shortlist=None)
        ),
        'cl-6460':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/pingolh-hardnet/hardnet68ds.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':74.3}, model_shortlist=None)
        ),
        'cl-6440':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/pingolh-hardnet/hardnet68.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.5}, model_shortlist=None)
        ),
        'cl-6450':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/pingolh-hardnet/hardnet85.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':78.0}, model_shortlist=None)
        ),
        #################pycls regnetx models#########################
        # pycls: classification regnetx200mf_224x224 expected_metric: 68.9% top-1 accuracy
        'cl-6360':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(reverse_channels=True),
            session=onnx_session_type(**sessions.get_onnx_bgr_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/fbr-pycls/regnetx-200mf.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':68.9}, model_shortlist=1)
        ),
        #################torchvision models#########################
        # torchvision: classification regnetx400mf_224x224 expected_metric: 72.834% top-1 accuracy
        'cl-6160':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/regnet_x_400mf_tv.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.834}, model_shortlist=20)
        ),
        # torchvision: classification regnetx800mf_224x224 expected_metric: 75.212% top-1 accuracy
        'cl-6170':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/regnet_x_800mf_tv.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.212}, model_shortlist=20)
        ),
        # pycls: classification regnetx1.6gf_224x224 expected_metric: 77.040% top-1 accuracy
        'cl-6180':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=True),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/regnet_x_1_6gf_tv.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':77.040}, model_shortlist=40)
        ),
        #################################################################
        #       TFLITE MODELS
        ##################tensorflow models##############################
        # mlperf/tf1 model: classification mobilenet_v1_224x224 expected_metric: 71.676 top-1 accuracy
        'cl-0000':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/mlperf/mobilenet_v1_1.0_224.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':71.676}, model_shortlist=10)
        ),
        # mlperf/tf-edge model: classification mobilenet_edgetpu_224 expected_metric: 75.6% top-1 accuracy
        'cl-0080':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/mlperf/mobilenet_edgetpu_224_1.0_float.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':75.6}, model_shortlist=40)
        ),
        # mlperf model: classification resnet50_v1.5 expected_metric: 76.456% top-1 accuracy
        'cl-0160':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.28, 103.53), input_scale=(1.0, 1.0, 1.0)),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/mlperf/resnet50_v1.5.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':76.456}, model_shortlist=30)
        ),
        #########################tensorflow1.0 models##################################
        # tensorflow/models: classification mobilenetv2_224x224 quant expected_metric: 70.0% top-1 accuracy
        'cl-0218':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite_quant(),
            session=tflite_session_type(**sessions.get_tflite_quant_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224_quant.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':70.0}, model_shortlist=None)
        ),
        # tensorflow/models: classification mobilenetv2_224x224 expected_metric: 71.9% top-1 accuracy
        'cl-0010':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v2_1.0_224.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':71.9}, model_shortlist=30)
        ),
        # tf hosted models: classification squeezenet_1 expected_metric: 57.5% top-1 accuracy
        'cl-0020':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_mean=(123.68, 116.78, 103.94), input_scale=(1/255, 1/255, 1/255)),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=True),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/squeezenet.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':57.5}, model_shortlist=None)
        ),
        # tensorflow/models: classification mobilenetv2_224x224 expected_metric: 75.0% top-1 accuracy
        'cl-0200':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v2_float_1.4_224.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':75.0}, model_shortlist=90)
        ),
        # tf hosted models: classification inception_v1_224_quant expected_metric: 69.63% top-1 accuracy
        'cl-0038':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite_quant(),
            session=tflite_session_type(**sessions.get_tflite_quant_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=True),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/inception_v1_224_quant.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':69.63}, model_shortlist=90)
        ),
        # tf hosted models: classification inception_v3 expected_metric: 78% top-1 accuracy
        'cl-0040':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(342, 299),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=True),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/inception_v3.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':78.0}, model_shortlist=90)
        ),
        # tf hosted models: classification mnasnet expected_metric: 74.08% top-1 accuracy
        'cl-0070':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mnasnet_1.0_224.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':74.08}, model_shortlist=None)
        ),
        # tf1 models: classification resnet50_v1 expected_metric: 75.2% top-1 accuracy
        'cl-0050':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.28, 103.53), input_scale=(1.0, 1.0, 1.0)),
                runtime_options=settings.runtime_options_tflite_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/resnet50_v1.tflite'),
            model_info=dict(metric_reference={'accuracy_top1%':75.2}, model_shortlist=None)
        ),
        # TODO: is this model's input correct? shouldn't it be 299 according to the slim page?
        # tf1 models: classification resnet50_v2 expected_metric: 75.6% top-1 accuracy
        'cl-0060':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/resnet50_v2.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':75.6}, model_shortlist=None)
        ),
        # tensorflow/models: classification mobilenet_v3-large-minimalistic_224_1.0_float expected_metric: 72.3% top-1 accuracy
        'cl-0260':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v3-large-minimalistic_224_1.0_float.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':72.3}, model_shortlist=None)
        ),
        # tensorflow/models: classification mobilenet_v3-small-minimalistic_224_1.0_float expected_metric: 61.9% top-1 accuracy
        'cl-0270':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v3-small-minimalistic_224_1.0_float.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':61.9}, model_shortlist=None)
        ),
        #################efficinetnet & tpu models#########################
        # tensorflow/tpu: classification efficinetnet-lite0_224x224 expected_metric: 75.1% top-1 accuracy
        'cl-0130':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-lite0-fp32.tflite'),
            model_info=dict(metric_reference={'accuracy_top1%':75.1}, model_shortlist=30)
        ),
        # tensorflow/tpu: classification efficinetnet-lite1_240x240 expected_metric: 76.7% top-1 accuracy
        'cl-0170':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(274, 240),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-lite1-fp32.tflite'),
            model_info=dict(metric_reference={'accuracy_top1%':76.7}, model_shortlist=90)
        ),
        # tensorflow/tpu: classification efficinetnet-lite4_300x300 expected_metric: 81.5% top-1 accuracy
        'cl-0140':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(343, 300),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=True),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-lite4-fp32.tflite'),
            model_info=dict(metric_reference={'accuracy_top1%':81.5}, model_shortlist=40)
        ),
        # tensorflow/tpu: classification efficientnet-edgetpu-S expected_metric: 77.23% top-1 accuracy
        'cl-0090':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=True),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-edgetpu-S_float.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':77.23}, model_shortlist=40)
        ),
        # tensorflow/tpu: classification efficientnet-edgetpu-M expected_metric: 78.69% top-1 accuracy
        'cl-0100':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(274, 240),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=True),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-edgetpu-M_float.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':78.69}, model_shortlist=90)
        ),
        # tensorflow/tpu: classification efficientnet-edgetpu-L expected_metric: 80.62% top-1 accuracy
        'cl-0190':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(343, 300),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=True),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-edgetpu-L_float.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':80.62}, model_shortlist=None)
        ),
        ###################################################################
        # complied for TVM - this model is repeated here and hard-coded to use tvmdlr session to generate an example tvmdlr artifact
        # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy
        'cl-3090':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=sessions.TVMDLRSession(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.88}, model_shortlist=10)
        ),
        # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy, QAT: 71.31%
        'cl-3098':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=sessions.TVMDLRSession(**sessions.get_onnx_quant_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_qat-p2.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.31}, model_shortlist=None)
        ),
        # torchvision: classification resnet50_224x224 expected_metric: 76.15% top-1 accuracy
        'cl-3110':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=sessions.TVMDLRSession(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/resnet50.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        # tensorflow/models: classification mobilenetv1_224x224 expected_metric: 71.0% top-1 accuracy (or is it 71.676% as this seems same as mlperf model)
        'cl-3520':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=sessions.TVMDLRSession(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':71.0}, model_shortlist=None)
        ),
        # tf1 models: classification resnet50_v1 expected_metric: 75.2% top-1 accuracy
        'cl-3530':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=sessions.TVMDLRSession(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.28, 103.53), input_scale=(1.0, 1.0, 1.0)),
                runtime_options=settings.runtime_options_tflite_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/resnet50_v1.tflite'),
            model_info=dict(metric_reference={'accuracy_top1%':75.2}, model_shortlist=None)
        ),
        
        ##################################################
        # CUSTOM MODELS
        ##################################################
        'pha-cl-0001':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=224, crop=224, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(127.5, 127.5, 127.5), 
                    input_scale=(1/127.5, 1/127.5, 1/127.5)),
                # runtime_options=settings.runtime_options_onnx_p2(),
                runtime_options=settings.runtime_options_onnx_p2(ext_options={'onnxruntime:graph_optimization_level': ort.GraphOptimizationLevel.ORT_DISABLE_ALL}),
                # runtime_options=settings.runtime_options_onnx_qat_v1(),
                # model_path='models/classification/vit_age_classification/vit_age_rgb_edit.onnx'),
                # model_path='models/classification/vit_age_classification/vit_age_rgb.onnx'),
                model_path='models/classification/vit_age_classification/vit_age_rgb_sim.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-cl-0002':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=224, crop=224, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(127.5, 127.5, 127.5), 
                    input_scale=(1/127.5, 1/127.5, 1/127.5)),
                runtime_options=settings.runtime_options_onnx_qat_v2(),
                model_path='models/classification/vit_age_classification/quantized_vit_rgb_quantized.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'rg-0001':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(
                resize=120, crop=120, data_layout=constants.NCHW, 
                reverse_channels=True, backend='cv2', interpolation=cv2.INTER_LINEAR, 
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=sessions.TVMDLRSession(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(127.5, 127.5, 127.5), 
                    input_scale=(0.0078125, 0.0078125, 0.0078125)
                    ),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path='/home/giang/Downloads/3D_landmarks_68_mb1_120x120.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'rg-0001':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(
                resize=120, crop=120, data_layout=constants.NCHW, 
                reverse_channels=True, backend='cv2', interpolation=cv2.INTER_LINEAR, 
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=sessions.TVMDLRSession(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(127.5, 127.5, 127.5), 
                    input_scale=(0.0078125, 0.0078125, 0.0078125)
                    ),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path='/home/giang/Downloads/3D_landmarks_68_mb1_120x120.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-hg-0001':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHAHG,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(0, 0, 0), 
                    input_scale=(1/255., 1/255., 1/255.)),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/hand_gestures/PHA_HandGestures_dev_mobilenet_v2_V6.0.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-hg-0002':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHAHG,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                # resize=400, crop=400, data_layout=constants.NCHW, 
                resize=224, crop=224, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(0, 0, 0), 
                    input_scale=(1/255., 1/255., 1/255.)),
                runtime_options=settings.runtime_options_onnx_p2(),
                # model_path='models/hand_gestures/PHA_HandGestures_dev_mobilenet_v2_V6.0.onnx'),
                # model_path='models/facial_landmark/drowsiness_r18_test.onnx'),
                # model_path='models/hand_gestures/test_model.onnx'),
                # model_path='models/facial_landmark/regression/resnet50.onnx'),
                model_path='models/facial_landmark/regression/resnet50_output24.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-hg-0003':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHAHG,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(0, 0, 0), 
                    input_scale=(1/255., 1/255., 1/255.)),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_qat_v2_p2(**quant_params_proto_path_disable_option),
                    {
                        'advanced_options:prequantized_model': '1',
                        'accuracy_level': '0',
                        # 'advanced_options:high_resolution_optimization': 0,
                        # 'advanced_options:pre_batchnorm_fold': 1,
                        # 'advanced_options:quantization_scale_type': 4,
                        # # 'advanced_options:activation_clipping': 1,
                        # # 'advanced_options:weight_clipping': 1,
                        # # advanced_options:bias_calibration: 1,
                        # # advanced_options:output_feature_16bit_names_list: '',
                        # # advanced_options:params_16bit_names_list: '',
                        # 'advanced_options:add_data_convert_ops': 3,
                        # 'debug_level': 3
                    }),
                model_path='models/hand_gestures/PHA_HandGestures_torch_quant.onnx'),
                # model_path='models/hand_gestures/PHA_HandGestures_torch_quant_editted.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-hg-0004':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHAHG,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(0, 0, 0), 
                    input_scale=(1/255., 1/255., 1/255.)),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/hand_gestures/PHA_HandGestures_dev_mobilenet_v2_V1.0_new_V1.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-hg-0005':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHAHG,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(0, 0, 0), 
                    input_scale=(1/255., 1/255., 1/255.)),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/hand_gestures/new/PHA_HandGestures_dev_mobilenet_v2_V1.0_new_V1.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-hg-0006':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHAHG,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(0, 0, 0), 
                    input_scale=(1/255., 1/255., 1/255.)),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                    # {'debug_level': 3},
                ),
                model_path='models/hand_gestures/new/PHA_HandGestures_dev_mobilenet_v2_V1.0_new_V1_copy.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-hg-0007':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHAHG,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(0, 0, 0), 
                    input_scale=(1/255., 1/255., 1/255.)),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                    # {'debug_level': 3},
                ),
                model_path='models/hand_gestures/new/Hand_gesture_2025_02_12_14.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-hg-0008':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHAHG,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                    # input_mean=(0, 0, 0), 
                    # input_scale=(1/255., 1/255., 1/255.)),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                    # {'debug_level': 3},
                ),
                model_path='models/hand_gestures/Hoa_test/0312_V7/handgesture_timm_mobilenetv4_conv_small_v1.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-hg-0009':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHAHG,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                    # input_mean=(0, 0, 0), 
                    # input_scale=(1/255., 1/255., 1/255.)),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                    # {'debug_level': 3},
                ),
                model_path='models/hand_gestures/Hoa_test/0313_V8/handgesture_torchvision_mobilenetv2_v1.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-hg-0010':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHAHG,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                    # input_mean=(0, 0, 0), 
                    # input_scale=(1/255., 1/255., 1/255.)),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                    # {'debug_level': 3},
                ),
                model_path='models/hand_gestures/Hoa_test/0428_V11/handgesture_timm_mobilenetv4_conv_small_e2400_r224_in1k_albumentations_ema.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-hg-testing':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHAHG,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHAHG]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=416, crop=416, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir,
                    input_mean=(0, 0, 0), 
                    input_scale=(1/255., 1/255., 1/255.)),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                    # {'debug_level': 3},
                ),
                # model_path = 'models/hand_gestures/testing/400_400/timm_mobilenetv2_100_classifier.onnx'),
                # model_path = 'models/hand_gestures/testing/400_400/timm_mobilenetv2_110d_classifier.onnx'),
                # model_path = 'models/hand_gestures/testing/400_400/timm_mobilenetv4_conv_small_classifier.onnx'),
                # model_path = 'models/hand_gestures/testing/416_416/damoyolo_ns_classifier.onnx'),
                model_path = 'models/hand_gestures/testing/416_416/object365_hgnetv2b0_classifier.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        # 
        'pha-gaze-0001':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=448, crop=448, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                # model_path='models/gaze/PHA_GazeEstimation_dev_mobilenetv4_ada_IR_V1.onnx'),
                # model_path='models/gaze/PHA_GazeEstimation_dev_mobilenetv4_IR_V1.onnx'),
                model_path='models/gaze/PHA_GazeEstimation_dev_mobilenetv4_IR_V1_merge_output.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-gaze-test':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=448, crop=448, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                # model_path='models/gaze/PHA_GazeEstimation_dev_mobilenetv4_ada_IR_V1.onnx'),
                # model_path='models/gaze/PHA_GazeEstimation_dev_mobilenetv4_IR_V1.onnx'),
                model_path='models/gaze/PHA_GazeEstimation_dev_mobilenetv4_IR_V1_merge_output.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-0001':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=256, crop=256, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                # runtime_options=settings.runtime_options_onnx_p2(ext_options={'onnxruntime:graph_optimization_level': ort.GraphOptimizationLevel.ORT_ENABLE_ALL}),
                runtime_options=settings.runtime_options_onnx_p2(),
                # model_path='models/facial_landmark/drowsiness_r18_12lmk_5nb_v0_opset11_editted.onnx'),
                # model_path='models/facial_landmark/drowsiness_r18_12lmk_5nb_v0_opset11_editted_opset11.onnx'),
                # model_path='models/facial_landmark/drowsiness_r18_Giang_custom.onnx'),
                # model_path='models/facial_landmark/drowsiness_r18_Giang_custom_optim.onnx'),
                # model_path='models/facial_landmark/drowsiness_Relu.onnx'),
                model_path='models/facial_landmark/drowsiness_Relu_editted.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-0002':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=256, crop=256, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/facial_landmark/drowsiness_r18_12lmk_5nb_v1_opset11_LRelu_editted.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-0003':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=256, crop=256, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                # model_path='models/facial_landmark/drowsiness_r18_test.onnx'),
                # model_path='models/facial_landmark/drowsiness_r18_test_editted.onnx'),
                model_path='models/facial_landmark/drowsiness_r18_12lmk_5nb_v3_opset11_RELU_no-post-proc.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-0004':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=256, crop=256, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                    {'debug_level': 3}
                ),
                # model_path='models/facial_landmark/regression/drowsiness_resnet_regression_2912.onnx'),
                # model_path='models/facial_landmark/regression/drowsiness_resnet_regression_2912_editted.onnx'),
                model_path='models/facial_landmark/regression/drowsiness_resnet_regression_2912_editted_maxpool.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-test':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    # settings.runtime_options_onnx_p2(),
                    {
                        'accuracy_level': 0,
                        # 'debug_level': 3
                    }
                ),
                # model_path = 'models/facial_landmark/regression/resnet50_output24.onnx'),
                # model_path = 'models/facial_landmark/regression/drowsiness_resnet_regression_2912_editted_maxpool_remove_BN.onnx'),
                # model_path = 'models/facial_landmark/regression/drowsiness_resnet_regression_2912.onnx'),
                # model_path = 'models/facial_landmark/regression/drowsiness_resnet_regression_2912_adapt_avg.onnx'),
                # model_path = 'models/facial_landmark/regression/drowsiness_resnet_regression_2912_adapt_avg.onnx'),
                model_path = 'models/hand_gestures/testing/400_400/timm_mobilenetv2_100_classifier.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-tfl':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_tflite(
                resize=256, crop=256, data_layout=constants.NHWC, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, pad_color=0),
            session=tflite_session_type(
                **sessions.get_tflite_session_cfg(
                    settings, 
                    work_dir=work_dir
                ),
                runtime_options=utils.dict_update(
                    settings.runtime_options_tflite_np2(),
                    {
                        'accuracy_level': 0,
                        # 'debug_level': 3
                    }
                ),
                model_path = 'models/facial_landmark/regression/converted_model.tflite'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        ###########################
        'pha-facelm-model-0001':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                    {
                        'debug_level': 3,
                    }
                ),
                model_path='models/facial_landmark/newest/model1_edited.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0002':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                    {
                        # 'debug_level': 3
                    }
                ),
                # model_path='models/facial_landmark/newest/model2_edited_new.onnx'),
                model_path='models/facial_landmark/newest/model2_onnxsim.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0003':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/model3_edited.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0004':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/model4_edited.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0005':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/model5_edited.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0006':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                    {
                        'debug_level': 3,
                    }
                ),
                # model_path='models/facial_landmark/newest/model6_edited.onnx'),
                # model_path='models/facial_landmark/newest/model6.onnx'),
                model_path='models/facial_landmark/newest/model6_edited_new.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0007':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/model7.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0008':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/model8.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0009':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/model9.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0010':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/model10.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0011':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/model11.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0012':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                    # {
                    #     'debug_level': 3
                    # }
                ),
                model_path='models/facial_landmark/newest/model12_edited_mul.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0013':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                    # {
                    #     'debug_level': 3
                    # }
                ),
                model_path='models/facial_landmark/newest/drowsiness_2025_01_08_11.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0014':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/regression/drowsiness_2025_02_10_11.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-model-0015':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/drowsiness_2025_02_14_11.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-edge-tv-0001':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/edgeai-tv/edgeai_mobilenetv2_192_192.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-edge-tv-0002':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=224, crop=224, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/edgeai-tv/new-edgeai_lab_mobilenet_v2_20191224.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-tv-0001':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=224, crop=224, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/edgeai-tv/modern_mobilenetv2.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-timm-0001':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=224, crop=224, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/edgeai-tv/timm_mobilenetv2_035.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-facelm-timm-0002':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=224, crop=224, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/facial_landmark/newest/edgeai-tv/timm_mobilenetv2_110d.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        ##############################
        'pha-driver-bh-0001':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=224, crop=224, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                # model_path='models/driver_behavior/mobilenetv3_large_behavior.onnx'),
                model_path='models/driver_behavior/mobilenetv3_large_behavior_editted.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-driver-bh-0002':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                # 'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=224, crop=224, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/driver_behavior/driver_behavior_2024_12_05_01.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-driver-bh-0003':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                # 'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=224, crop=224, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/driver_behavior/driver_behavior_2024_12_09_02.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-driver-bh-0004':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                # 'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=224, crop=224, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/driver_behavior/driver_behavior_2024_12_11_03.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-driver-bh-0005':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                # 'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/driver_behavior/kwanju/PHA_Driver_Behavior_dev_mobilenet_v2_V1.0.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-driver-bh-0006':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                # 'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/driver_behavior/kwanju/PHA_Driver_Behavior_dev_mobilenet_v2_V2.0.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-driver-bh-0007':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                # 'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/driver_behavior/kwanju/PHA_Driver_Behavior_dev_mobilenet_v2_V3.0.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-driver-bh-0008':utils.dict_update( # different dataset
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                # 'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=400, crop=400, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/driver_behavior/kwanju/PHA_Driver_Behavior_dev_mobilenet_v2_V3.0.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-driver-bh-0009':utils.dict_update( # different dataset
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                # 'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=448, crop=448, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/driver_behavior/kwanju/PHA_Driver_Behavior_dev_mobilenet_v2_V4.0.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-driver-bh-0010':utils.dict_update( # different dataset
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                # 'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=448, crop=448, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/driver_behavior/kwanju/PHA_Driver_Behavior_dev_mobilenet_v2_V5.0.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-driver-bh-0011':utils.dict_update( # different dataset
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                'postprocess': postproc_transforms.get_transform_classification()
                # 'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=448, crop=448, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path='models/driver_behavior/kwanju/PHA_Driver_Behavior_dev_mobilenet_v2_V6.0.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        #########################################################
        'kadif-3d-0001':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=(800, 1200), crop=(800, 1200), data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/mono3dDetection/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_tidl.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        ###############################################start#################################################
        'pha-2-facelm-24-viz-0001':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/face_landmark_24keypoints_viz/2025_07_15_02.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-2-facelm-24-viz-0002':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/face_landmark_24keypoints_viz/facelmk24_0827.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-2-facelm-24-viz-0003':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/face_landmark_24keypoints_viz/facelmk24_0827_crop_256.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha-2-facelm-24-viz-0004':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/face_landmark_24keypoints_viz/facelmk24_0827_v3.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
         'pha-2-facelm-24-viz-0006':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=192, crop=192, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=utils.dict_update(
                    settings.runtime_options_onnx_p2(),
                ),
                model_path='models/face_landmark_24keypoints_viz/best_nme_best_v3.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha2-gaze-0001':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=96, crop=96, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                # model_path='models/gaze/PHA_GazeEstimation_dev_mobilenetv4_ada_IR_V1.onnx'),
                # model_path='models/gaze/PHA_GazeEstimation_dev_mobilenetv4_IR_V1.onnx'),
                model_path='models/gaze_model/vgg16_v1/vgg16_test.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),
        'pha2-gaze-0002':utils.dict_update(
            {
                'task_type': 'classification',
                'dataset_category': datasets.DATASET_CATEGORY_PHACLS,
                'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['calibration_dataset'],
                 'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_PHACLS]['input_dataset'],
                # 'postprocess': postproc_transforms.get_transform_classification()
                'postprocess': postproc_transforms.get_transform_none()
                },
            preprocess=preproc_transforms.get_transform_onnx(
                resize=96, crop=96, data_layout=constants.NCHW, 
                reverse_channels=False, backend='cv2',
                resize_with_pad=False, add_flip_image=False, pad_color=0),
            session=onnx_session_type(
                **sessions.get_onnx_session_cfg(
                    settings, 
                    work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                # model_path='models/gaze/PHA_GazeEstimation_dev_mobilenetv4_ada_IR_V1.onnx'),
                # model_path='models/gaze/PHA_GazeEstimation_dev_mobilenetv4_IR_V1.onnx'),
                model_path='models/gaze_model/regnet/0829_gaze.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None)
        ),

    }
    return pipeline_configs


