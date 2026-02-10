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

import warnings

from .onnx_backend_dataset import *
from .image_cls import *
from .image_seg import *
from .image_det import *
from .image_reg import *

from .coco_det import *
from .coco_seg import *
from .imagenet import *
from .imagenetv2 import *
from .pha_cls import *
from .pha_cls_hg import *
from .regression_gaze import *
from .regression_facelm import *
from .aflw import *
from .cityscapes import *
from .ade20k import *
from .voc_seg import *
from .nyudepthv2 import *
from .ycbv import *
from .modelmaker_datasets import *

from .coco_kpts import *
from .widerface_det import *

from .robokit_seg import *
from .robokit_visloc import *

from .kitti_2015 import *

try:
    from .kitti_lidar_det import KittiLidar3D
except ImportError as e:
    warnings.warn(f'kitti_lidar_det could not be imported - {str(e)}')
    KittiLidar3D = None

DATASET_CATEGORY_ICMS_DET = 'icms_det' # sami icms dataset 


DATASET_CATEGORY_IMAGENET = 'imagenet'
DATASET_CATEGORY_PHACLS = 'phacls'
DATASET_CATEGORY_PHAHG = 'phahg'
DATASET_CATEGORY_COCO = 'coco'
DATASET_CATEGORY_PHA = 'pha'
DATASET_CATEGORY_AFLW = 'aflw'
DATASET_CATEGORY_WIDERFACE = 'widerface'
DATASET_CATEGORY_ADE20K32 = 'ade20k32'
DATASET_CATEGORY_ADE20K = 'ade20k'
DATASET_CATEGORY_VOC2012 = 'voc2012'
DATASET_CATEGORY_COCOSEG21 = 'cocoseg21'
DATASET_CATEGORY_COCOKPTS = 'cocokpts'
DATASET_CATEGORY_PHAKPTS = 'phakpts'
DATASET_CATEGORY_NYUDEPTHV2 = 'nyudepthv2'
DATASET_CATEGORY_CITYSCAPES = 'cityscapes'
DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD = 'ti-robokit_semseg_zed1hd'
DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD = 'ti-robokit_visloc_zed1hd'
DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS = 'kitti_lidar_det_1class'
DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS = 'kitti_lidar_det_3class'
DATASET_CATEGORY_KITTI_2015 = 'kitti_2015'
DATASET_CATEGORY_YCBV = 'ycbv'

DATASET_CATEGORY_REGRESSION_GAZE = 'regression_gaze'
DATASET_CATEGORY_REGRESSION_FACELM = 'regression_facelm'
dataset_info_dict = {
    #------------------------image classification datasets--------------------------#
    # Original ImageNet
    'imagenet':{'task_type':'classification', 'category':DATASET_CATEGORY_IMAGENET, 'type':ImageNetCls, 'size':50000, 'split':'val'},
    'imagenetv1':{'task_type':'classification', 'category':DATASET_CATEGORY_IMAGENET, 'type':ImageNetCls, 'size':50000, 'split':'val'},
    'phacls': {'task_type':'classification', 'category': DATASET_CATEGORY_PHACLS, 'type': PHACls, 'size': 36},
    'phahg': {'task_type':'classification', 'category': DATASET_CATEGORY_PHAHG, 'type': PHAhg, 'size': 6760},

    #AFLW Regression
    'aflw':{'task_type':'classification', 'category':DATASET_CATEGORY_AFLW, 'type':AFLWReg, 'size': 2000, 'split':'val'},
    # ImageNetV2 as explained in imagenet_v2.py
    'imagenetv2c':{'task_type':'classification', 'category':DATASET_CATEGORY_IMAGENET, 'type':ImageNetV2C, 'size':10000, 'split':'val'},
    'imagenetv2b':{'task_type':'classification', 'category':DATASET_CATEGORY_IMAGENET, 'type':ImageNetV2B, 'size':10000, 'split':'val'},
    'imagenetv2a':{'task_type':'classification', 'category':DATASET_CATEGORY_IMAGENET, 'type':ImageNetV2A, 'size':10000, 'split':'val'},
    #------------------------object detection datasets--------------------------#
    'coco': {'task_type':'detection', 'category':DATASET_CATEGORY_COCO, 'type':COCODetection, 'size':5000, 'split':'val2017'},
    'pha': {'task_type': 'detection', 'category': DATASET_CATEGORY_PHA, 'type': COCODetection, 'size': 10000, 'split': 'images'},



     # sami
    'icms_det': {'task_type':'detection', 'category':DATASET_CATEGORY_ICMS_DET, 'type':COCODetection, 'size': 10000, 'split':'images'},





    'widerface': {'task_type':'detection', 'category':DATASET_CATEGORY_WIDERFACE, 'type':WiderFaceDetection, 'size':3226, 'split':'val'},
    #------------------------semantic segmentation datasets--------------------------#
    'ade20k32': {'task_type':'segmentation', 'category':DATASET_CATEGORY_ADE20K32, 'type':ADE20KSegmentation, 'size':2000, 'split':'validation'},
    'ade20k': {'task_type':'segmentation', 'category':DATASET_CATEGORY_ADE20K, 'type':ADE20KSegmentation, 'size':2000, 'split':'validation'},
    'voc2012': {'task_type':'segmentation', 'category':DATASET_CATEGORY_VOC2012, 'type':VOC2012Segmentation, 'size':1449, 'split':'val'},
    'cocoseg21': {'task_type':'segmentation', 'category':DATASET_CATEGORY_COCOSEG21, 'type':COCOSegmentation, 'size':5000, 'split':'val2017'},
    'ti-robokit_semseg_zed1hd': {'task_type':'segmentation', 'category':DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD, 'type':RobokitSegmentation, 'size':49, 'split':'val'},
    'ti-robokit_visloc_zed1hd': {'task_type':'visual_localization', 'category':DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD, 'type':RobokitVisualLocalization, 'size':49, 'split':'val'},
    #------------------------human pose estimation datasets--------------------------#
    'cocokpts': {'task_type':'keypoint_detection', 'category':DATASET_CATEGORY_COCOKPTS, 'type':COCOKeypoints, 'size':5000, 'split':'val2017'},
    'phakpts': {'task_type':'keypoint_detection', 'category':DATASET_CATEGORY_PHAKPTS, 'type':COCOKeypoints, 'size':5000, 'split':'val2017'},
    #------------------------depth estimation datasets--------------------------#
    'nyudepthv2': {'task_type':'depth_estimation', 'category':DATASET_CATEGORY_NYUDEPTHV2, 'type':NYUDepthV2, 'size':654, 'split':'val'},
    #------------------------object 6d pose estimation datasets--------------------------#
    'ycbv': {'task_type':'object_6d_pose_estimation', 'category':DATASET_CATEGORY_YCBV, 'type': YCBV, 'size':900, 'split':'test'},
    #------------------------regression datasets--------------------------#
    'regression_gaze': {'task_type':'gaze_estimation', 'category': DATASET_CATEGORY_REGRESSION_GAZE, 'type': RegressionGazeDataset, 'size': 15000},
    'regression_facelm': {'task_type':'face_landmark_24', 'category': DATASET_CATEGORY_REGRESSION_FACELM, 'type': RegressionFacelmDataset, 'size': 15000},
 }


dataset_info_dict_experimental = {
    #------------------------semantic segmentation datasets--------------------------#
    'cityscapes': {'task_type':'segmentation', 'category':DATASET_CATEGORY_CITYSCAPES, 'type':CityscapesSegmentation, 'size':500, 'split':'val'},
    #------------------------3D OD datasets--------------------------#
    'kitti_lidar_det_1class': {'task_type':'3d-detection', 'category':DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS, 'type':KittiLidar3D, 'size':3769, 'split':'val'},
    'kitti_lidar_det_3class': {'task_type': '3d-detection', 'category': DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS,'type': KittiLidar3D, 'size': 3769, 'split': 'val'},
    #----------------------- Stereo disparity datasets--------------------------#
    'kitti_2015': {'task_type':'stereo-disparity', 'category':DATASET_CATEGORY_KITTI_2015, 'type':Kitti2015, 'size':159, 'split':'training'},
}


def get_dataset_info_dict(settings):
    dset_info_dict = dataset_info_dict.copy()
    if settings is not None and settings.experimental_models:
        dset_info_dict.update(dataset_info_dict_experimental)
    #
    return dset_info_dict


def get_dataset_categories(settings=None, task_type=None):
    dset_info_dict = get_dataset_info_dict(settings)
    # we are picking category instead of the actual dataset name/variant.
    # the actual dataset to be used cal is selected in get_dataset()
    if task_type is not None:
        dataset_names = [value['category'] for key,value in dset_info_dict.items() if value['task_type'] == task_type]
    else:
        dataset_names = [value['category'] for key,value in dset_info_dict.items()]
    #
    # make it unique - set() is unordered - so use dict.fromkeys()
    dataset_categories = list(dict.fromkeys(dataset_names).keys())
    return dataset_categories


def get_dataset_names(settings, task_type=None):
    print(utils.log_color('WARNING', 'name change', f'please use datasets.get_dataset_categories() '
                                                    f'instead of datasets.get_dataset_names()'))
    dataset_categories = get_dataset_categories(settings, task_type)
    return dataset_categories


def _initialize_datasets(settings):
    dataset_categories = get_dataset_categories(settings)
    dataset_cache = {
        ds_category: {'calibration_dataset':ds_category, 'input_dataset':ds_category} \
        for ds_category in dataset_categories
    }
    return dataset_cache


def get_datasets(settings, download=False, dataset_list=None):
    dataset_cache = _initialize_datasets(settings)
    dset_info_dict = get_dataset_info_dict(settings)
    dataset_list = dataset_list or get_dataset_categories(settings)


################################################# sami ################################################
    if check_dataset_load(settings, DATASET_CATEGORY_ICMS_DET) and (DATASET_CATEGORY_ICMS_DET in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_ICMS_DET} variant:{DATASET_CATEGORY_ICMS_DET}"))

        icms_det_calib_cfg = dict(
            num_classes = 10,
            image_dir=f'{settings.datasets_path}/icms_det/images',
            annotation_file=f'{settings.datasets_path}/icms_det/annotations/instances_val.json',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=DATASET_CATEGORY_ICMS_DET)
        
        icms_det_val_cfg = dict(
            num_classes = 10,
            image_dir=f'{settings.datasets_path}/icms_det/images',
            annotation_file=f'{settings.datasets_path}/icms_det/annotations/instances_val.json',
            shuffle=False, # can be set to True as well, if needed
            # num_frames=min(settings.num_frames,5000),
            num_frames=settings.num_frames,
            name=DATASET_CATEGORY_ICMS_DET)
        dataset_cache[DATASET_CATEGORY_ICMS_DET]['calibration_dataset'] = COCODetection(**icms_det_calib_cfg, download=False)
        dataset_cache[DATASET_CATEGORY_ICMS_DET]['input_dataset'] = COCODetection(**icms_det_val_cfg, download=False)
################################################# sami ################################################

################################################# sami_icms2 ################################################
    if check_dataset_load(settings, DATASET_CATEGORY_ICMS_DET) and (DATASET_CATEGORY_ICMS_DET in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_ICMS_DET} variant:{DATASET_CATEGORY_ICMS_DET}"))

        icms_det_calib_cfg = dict(
            num_classes = 6,
            image_dir=f'{settings.datasets_path}/icms_det2/images',
            annotation_file=f'{settings.datasets_path}/icms_det2/annotations/instances_val.json',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=DATASET_CATEGORY_ICMS_DET)
        
        icms_det_val_cfg = dict(
            num_classes = 6,
            image_dir=f'{settings.datasets_path}/icms_det2/images',
            annotation_file=f'{settings.datasets_path}/icms_det2/annotations/instances_val.json',
            shuffle=False, # can be set to True as well, if needed
            # num_frames=min(settings.num_frames,5000),
            num_frames=settings.num_frames,
            name=DATASET_CATEGORY_ICMS_DET)
        dataset_cache[DATASET_CATEGORY_ICMS_DET]['calibration_dataset'] = COCODetection(**icms_det_calib_cfg, download=False)
        dataset_cache[DATASET_CATEGORY_ICMS_DET]['input_dataset'] = COCODetection(**icms_det_val_cfg, download=False)
################################################# sami_icms2 ################################################





    if check_dataset_load(settings, DATASET_CATEGORY_IMAGENET) and (DATASET_CATEGORY_IMAGENET in dataset_list):
        dataset_variant = settings.dataset_type_dict[DATASET_CATEGORY_IMAGENET] if \
            settings.dataset_type_dict is not None else DATASET_CATEGORY_IMAGENET
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_IMAGENET} variant:{dataset_variant}"))
        # dataset settings
        imagenet_dataset_dict = dset_info_dict[dataset_variant]
        ImageNetDataSetType = imagenet_dataset_dict['type']
        imagenet_split = imagenet_dataset_dict['split']
        num_imgs = imagenet_dataset_dict['size']
        # the cfg to be used to construct the dataset class
        imagenet_cls_calib_cfg = dict(
            path=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}',
            split=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}.txt',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=dataset_variant)
        imagenet_cls_val_cfg = dict(
            path=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}',
            split=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}.txt',
            shuffle=True,
            num_frames=min(settings.num_frames,num_imgs),
            name=dataset_variant)
        # what is provided is mechanism to select one of the imagenet variants
        # but only one is selected and assigned to the key imagenet
        # all the imagenet models will use this variant.
        print(f'Value of download here: {download}')# TODO: LUKE remove
        dataset_cache[DATASET_CATEGORY_IMAGENET]['calibration_dataset'] = ImageNetDataSetType(**imagenet_cls_calib_cfg, download=True)
        dataset_cache[DATASET_CATEGORY_IMAGENET]['input_dataset'] = ImageNetDataSetType(**imagenet_cls_val_cfg, download=True)
    #
    if check_dataset_load(settings, DATASET_CATEGORY_PHAHG) and (DATASET_CATEGORY_PHAHG in dataset_list):
        dataset_variant = settings.dataset_type_dict[DATASET_CATEGORY_PHAHG] if \
            settings.dataset_type_dict is not None else DATASET_CATEGORY_PHAHG
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_PHAHG} variant:{dataset_variant}"))
        # dataset settings
        phahg_dataset_dict = dset_info_dict[dataset_variant]
        phahgDataSetType = phahg_dataset_dict['type']
        # phacls_split = phacls_dataset_dict['split']
        num_imgs = phahg_dataset_dict['size']
        # the cfg to be used to construct the dataset class
        pha_hg_calib_cfg = dict(
            # image_path='dependencies/datasets/pha_handgesture/valid',
            # annotations_file='dependencies/datasets/pha_handgesture/valid.json',
            image_path='dependencies/datasets/pha_handgesture/new_data/images',
            annotations_file='dependencies/datasets/pha_handgesture/new_data/labels.json',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=dataset_variant)
        pha_hg_val_cfg = dict(
            # image_path='dependencies/datasets/pha_handgesture/valid',
            # annotations_file='dependencies/datasets/pha_handgesture/valid.json',
            image_path='dependencies/datasets/pha_handgesture/new_data/images',
            annotations_file='dependencies/datasets/pha_handgesture/new_data/labels.json',
            shuffle=True,
            num_frames=min(settings.num_frames,num_imgs),
            name=dataset_variant)
        # what is provided is mechanism to select one of the imagenet variants
        # but only one is selected and assigned to the key imagenet
        # all the imagenet models will use this variant.
        print(f'Value of download here: {download}')# TODO: LUKE remove
        dataset_cache[DATASET_CATEGORY_PHAHG]['calibration_dataset'] = phahgDataSetType(**pha_hg_calib_cfg, download=False)
        dataset_cache[DATASET_CATEGORY_PHAHG]['input_dataset'] = phahgDataSetType(**pha_hg_val_cfg, download=False)
    #
    if check_dataset_load(settings, DATASET_CATEGORY_PHACLS) and (DATASET_CATEGORY_PHACLS in dataset_list):
        dataset_variant = settings.dataset_type_dict[DATASET_CATEGORY_PHACLS] if \
            settings.dataset_type_dict is not None else DATASET_CATEGORY_PHACLS
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_PHACLS} variant:{dataset_variant}"))
        # dataset settings
        phacls_dataset_dict = dset_info_dict[dataset_variant]
        phaclsDataSetType = phacls_dataset_dict['type']
        # phacls_split = phacls_dataset_dict['split']
        num_imgs = phacls_dataset_dict['size']
        # the cfg to be used to construct the dataset class
        pha_cls_calib_cfg = dict(
            # image_path=f'dependencies/datasets/pha_age/images',
            # annotations_file=f'dependencies/datasets/pha_age/label.json',
            # image_path='dependencies/datasets/pha_gaze/test/Face',
            # annotations_file='dependencies/datasets/pha_gaze/labels.json',
            # image_path='dependencies/datasets/face_landmark/test',
            # annotations_file='dependencies/datasets/face_landmark/labels.json',
            # image_path='dependencies/datasets/pha_driver_behavior/images',
            # annotations_file='dependencies/datasets/pha_driver_behavior/labels.json',
            # image_path='dependencies/datasets/pha_driver_behavior/new_dataset/images',
            # annotations_file='dependencies/datasets/pha_driver_behavior/new_dataset/labels.json',
            # image_path='dependencies/datasets/pha_driver_behavior/kwanju_dataset/v1/images',
            # annotations_file='dependencies/datasets/pha_driver_behavior/kwanju_dataset/v1/labels.json',
            image_path='dependencies/datasets/pha2_facelm_24_viz/images',
            annotations_file='dependencies/datasets/pha2_facelm_24_viz/labels.json',
            # image_path='dependencies/datasets/gaze/vgg16_test/p02',
            # annotations_file='dependencies/datasets/gaze/vgg16_test/labels.json',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=dataset_variant)
        pha_cls_val_cfg = dict(
            # image_path=f'dependencies/datasets/pha_age/images',
            # annotations_file=f'dependencies/datasets/pha_age/label.json',
            # image_path='dependencies/datasets/pha_gaze/test/Face',
            # annotations_file='dependencies/datasets/pha_gaze/labels.json',
            # image_path='dependencies/datasets/face_landmark/test',
            # annotations_file='dependencies/datasets/face_landmark/labels.json',
            # image_path='dependencies/datasets/pha_driver_behavior/images',
            # annotations_file='dependencies/datasets/pha_driver_behavior/labels.json',
            # image_path='dependencies/datasets/pha_driver_behavior/new_dataset/images',
            # annotations_file='dependencies/datasets/pha_driver_behavior/new_dataset/labels.json',
            # image_path='dependencies/datasets/pha_driver_behavior/kwanju_dataset/v1/images',
            # annotations_file='dependencies/datasets/pha_driver_behavior/kwanju_dataset/v1/labels.json',
            image_path='dependencies/datasets/pha2_facelm_24_viz/images',
            annotations_file='dependencies/datasets/pha2_facelm_24_viz/labels.json',
            # image_path='dependencies/datasets/gaze/vgg16_test/p02',
            # annotations_file='dependencies/datasets/gaze/vgg16_test/labels.json',
            shuffle=True,
            num_frames=min(settings.num_frames,num_imgs),
            name=dataset_variant)
        # what is provided is mechanism to select one of the imagenet variants
        # but only one is selected and assigned to the key imagenet
        # all the imagenet models will use this variant.
        print(f'Value of download here: {download}')# TODO: LUKE remove
        dataset_cache[DATASET_CATEGORY_PHACLS]['calibration_dataset'] = phaclsDataSetType(**pha_cls_calib_cfg, download=False)
        dataset_cache[DATASET_CATEGORY_PHACLS]['input_dataset'] = phaclsDataSetType(**pha_cls_val_cfg, download=False)
    #
    if check_dataset_load(settings, DATASET_CATEGORY_REGRESSION_GAZE) and (DATASET_CATEGORY_REGRESSION_GAZE in dataset_list):
        dataset_variant = settings.dataset_type_dict[DATASET_CATEGORY_REGRESSION_GAZE] if \
            settings.dataset_type_dict is not None else DATASET_CATEGORY_REGRESSION_GAZE
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_REGRESSION_GAZE} variant:{dataset_variant}"))
        # dataset settings
        regression_dataset_dict = dset_info_dict[dataset_variant]
        regressionDataSetType = regression_dataset_dict['type']
        num_imgs = regression_dataset_dict['size']
        # the cfg to be used to construct the dataset class
        regression_calib_cfg = dict(
            image_path='dependencies/pha_2_datasets/gaze_Y_channel/images',
            annotations_file='dependencies/pha_2_datasets/gaze_Y_channel/labels.json',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=dataset_variant)
        regression_val_cfg = dict(
            image_path='dependencies/pha_2_datasets/gaze_Y_channel/images',
            annotations_file='dependencies/pha_2_datasets/gaze_Y_channel/labels.json',
            shuffle=True,
            num_frames=min(settings.num_frames,num_imgs),
            name=dataset_variant)
        # what is provided is mechanism to select one of the regression variants
        # but only one is selected and assigned to the key regression
        # all the regression models will use this variant.
        print(f'Value of download here: {download}')# TODO: LUKE remove
        dataset_cache[DATASET_CATEGORY_REGRESSION_GAZE]['calibration_dataset'] = regressionDataSetType(**regression_calib_cfg, download=False)
        dataset_cache[DATASET_CATEGORY_REGRESSION_GAZE]['input_dataset'] = regressionDataSetType(**regression_val_cfg, download=False)
    #
    
    if check_dataset_load(settings, DATASET_CATEGORY_REGRESSION_FACELM) and (DATASET_CATEGORY_REGRESSION_FACELM in dataset_list):
        dataset_variant = settings.dataset_type_dict[DATASET_CATEGORY_REGRESSION_FACELM] if \
            settings.dataset_type_dict is not None else DATASET_CATEGORY_REGRESSION_FACELM
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_REGRESSION_FACELM} variant:{dataset_variant}"))
        # dataset settings
        regression_dataset_dict = dset_info_dict[dataset_variant]
        regressionDataSetType = regression_dataset_dict['type']
        num_imgs = regression_dataset_dict['size']
        # the cfg to be used to construct the dataset class
        regression_calib_cfg = dict(
            image_path='dependencies/pha_2_datasets/facial_24/v1/images',
            annotations_file='dependencies/pha_2_datasets/facial_24/v1/labels.json',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=dataset_variant)
        regression_val_cfg = dict(
            image_path='dependencies/pha_2_datasets/facial_24/v1/images',
            annotations_file='dependencies/pha_2_datasets/facial_24/v1/labels.json',
            shuffle=True,
            num_frames=min(settings.num_frames,num_imgs),
            name=dataset_variant)
        # what is provided is mechanism to select one of the regression variants
        # but only one is selected and assigned to the key regression
        # all the regression models will use this variant.
        print(f'Value of download here: {download}')# TODO: LUKE remove
        dataset_cache[DATASET_CATEGORY_REGRESSION_FACELM]['calibration_dataset'] = regressionDataSetType(**regression_calib_cfg, download=False)
        dataset_cache[DATASET_CATEGORY_REGRESSION_FACELM]['input_dataset'] = regressionDataSetType(**regression_val_cfg, download=False)
    
    
    if check_dataset_load(settings, DATASET_CATEGORY_COCOKPTS) and (DATASET_CATEGORY_COCOKPTS in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_COCOKPTS} variant:{DATASET_CATEGORY_COCOKPTS}"))
        filter_imgs = True
        coco_kpts_calib_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=DATASET_CATEGORY_COCOKPTS,
            filter_imgs=filter_imgs)
        coco_kpts_val_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=False, #TODO: need to make COCODetection.evaluate() work with shuffle
            num_frames=min(settings.num_frames,5000),
            name=DATASET_CATEGORY_COCOKPTS,
            filter_imgs=filter_imgs)

        dataset_cache[DATASET_CATEGORY_COCOKPTS]['calibration_dataset'] = COCOKeypoints(**coco_kpts_calib_cfg, download=download)
        dataset_cache[DATASET_CATEGORY_COCOKPTS]['input_dataset'] = COCOKeypoints(**coco_kpts_val_cfg, download=False)
    #
    if check_dataset_load(settings, DATASET_CATEGORY_YCBV) and (DATASET_CATEGORY_YCBV in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_YCBV} variant:{DATASET_CATEGORY_YCBV}"))
        filter_imgs = True
        ycbv_calib_cfg = dict(
            path=f'{settings.datasets_path}/ycbv',
            split='test',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=DATASET_CATEGORY_YCBV,
            filter_imgs=filter_imgs)
        ycbv_val_cfg = dict(
            path=f'{settings.datasets_path}/ycbv',
            split='test',
            shuffle=False,
            num_frames=min(settings.num_frames,900),
            name=DATASET_CATEGORY_YCBV,
            filter_imgs=filter_imgs)

        dataset_cache[DATASET_CATEGORY_YCBV]['calibration_dataset'] = YCBV(**ycbv_calib_cfg, download=download)
        dataset_cache[DATASET_CATEGORY_YCBV]['input_dataset'] = YCBV(**ycbv_val_cfg, download=False)
    #
    if check_dataset_load(settings, DATASET_CATEGORY_COCO) and (DATASET_CATEGORY_COCO in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_COCO} variant:{DATASET_CATEGORY_COCO}"))
        coco_det_calib_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=DATASET_CATEGORY_COCO)
        coco_det_val_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=False, # can be set to True as well, if needed
            num_frames=min(settings.num_frames,5000),
            name=DATASET_CATEGORY_COCO)
        dataset_cache[DATASET_CATEGORY_COCO]['calibration_dataset'] = COCODetection(**coco_det_calib_cfg, download=True)
        dataset_cache[DATASET_CATEGORY_COCO]['input_dataset'] = COCODetection(**coco_det_val_cfg, download=True)
    
    if check_dataset_load(settings, DATASET_CATEGORY_PHA) and (DATASET_CATEGORY_PHA in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_PHA} variant:{DATASET_CATEGORY_PHA}"))
        pha_det_calib_cfg = dict(
            num_classes = 7,
            path=f'{settings.datasets_path}',
            split='images',
            # image_dir='dependencies/datasets/pha/6_classes_mini/images',
            # annotation_file='dependencies/datasets/pha/6_classes_mini/annotations/filtered_annotations.json',
            # image_dir='dependencies/datasets/pha/9_classes_mini/images',
            # annotation_file='dependencies/datasets/pha/9_classes_mini/annotations/combined_instances_fix_3.json',
            # image_dir='dependencies/datasets/pha_mid_demo/images',
            # annotation_file='dependencies/datasets/pha_mid_demo/cpd_classes_val.json',
            # image_dir='dependencies/datasets/pha_1024_cpd_od/images',
            # annotation_file='dependencies/datasets/pha_1024_cpd_od/final_v1_val.json',
            # image_dir='dependencies/datasets/pha_object_detection/1204_unseen_telelian_data/images',
            # annotation_file='dependencies/datasets/pha_object_detection/1204_unseen_telelian_data/final_v1.json',
            # image_dir='dependencies/datasets/pha_object_detection/2024_12_23_02/images',
            # annotation_file='dependencies/datasets/pha_object_detection/2024_12_23_02/final_v1.json',
            # image_dir='dependencies/datasets/pha_object_detection/2025_01_06_04/images',
            # annotation_file='dependencies/datasets/pha_object_detection/2025_01_06_04/final_v1.json',
            # image_dir='dependencies/datasets/pha_object_detection/2025_01_03_03/images',
            # annotation_file='dependencies/datasets/pha_object_detection/2025_01_03_03/final_v1.json',
            #############################pha2 od 7 classes##################################
            image_dir='dependencies/pha_2_datasets/detection/od_7classes/images',
            annotation_file='dependencies/pha_2_datasets/detection/od_7classes/annotations/7class.json',

            shuffle=True,
            num_frames=settings.calibration_frames,
            name=DATASET_CATEGORY_PHA)
        pha_det_val_cfg = dict(
            num_classes = 7,
            path=f'{settings.datasets_path}',
            split='images',
            # image_dir='dependencies/datasets/pha/6_classes_mini/images',
            # annotation_file='dependencies/datasets/pha/6_classes_mini/annotations/filtered_annotations.json',
            # image_dir='dependencies/datasets/pha/9_classes_mini/images',
            # annotation_file='dependencies/datasets/pha/9_classes_mini/annotations/combined_instances_fix_3.json',
            # image_dir='dependencies/datasets/pha_mid_demo/images',
            # annotation_file='dependencies/datasets/pha_mid_demo/cpd_classes_val.json',
            # image_dir='dependencies/datasets/pha_1024_cpd_od/images',
            # annotation_file='dependencies/datasets/pha_1024_cpd_od/final_v1_val.json',
            # image_dir='dependencies/datasets/pha_object_detection/1204_unseen_telelian_data/images',
            # annotation_file='dependencies/datasets/pha_object_detection/1204_unseen_telelian_data/final_v1.json',
            # image_dir='dependencies/datasets/pha_object_detection/2024_12_23_02/images',
            # annotation_file='dependencies/datasets/pha_object_detection/2024_12_23_02/final_v1.json',
            # image_dir='dependencies/pha_2_datasets/detection/od_7classes/images',
            # annotation_file='dependencies/pha_2_datasets/detection/od_7classes/annotations/7class.json',
            # image_dir='dependencies/datasets/pha_object_detection/2025_01_06_04/images',
            # annotation_file='dependencies/datasets/pha_object_detection/2025_01_06_04/final_v1.json',
            # image_dir='dependencies/datasets/pha_object_detection/2025_02_06_01/images/images_crop',
            # annotation_file='dependencies/datasets/pha_object_detection/2025_02_06_01/9classes_crop_resize_hod.json',
            #############################pha2 od 7 classes##################################
            image_dir='dependencies/pha_2_datasets/detection/od_7classes/images',
            annotation_file='dependencies/pha_2_datasets/detection/od_7classes/annotations/7class.json',
            shuffle=False, # can be set to True as well, if needed
            # num_frames=min(settings.num_frames,5000),
            num_frames=settings.num_frames,
            name=DATASET_CATEGORY_PHA)
        dataset_cache[DATASET_CATEGORY_PHA]['calibration_dataset'] = COCODetection(**pha_det_calib_cfg, download=False)
        dataset_cache[DATASET_CATEGORY_PHA]['input_dataset'] = COCODetection(**pha_det_val_cfg, download=False)
        
    if check_dataset_load(settings, DATASET_CATEGORY_PHAKPTS) and (DATASET_CATEGORY_PHAKPTS in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_PHAKPTS} variant:{DATASET_CATEGORY_PHAKPTS}"))
        pha_kpts_calib_cfg = dict(
            path=f'{settings.datasets_path}',
            split='images',
            # image_dir='dependencies/datasets/pha_kps/images_resize',
            # annotation_file='dependencies/datasets/pha_kps/annotaions/subset_instances_val2017_fix.json',
            # image_dir='dependencies/datasets/pha_kps_new/images',
            # annotation_file='dependencies/datasets/pha_kps_new/instances.json',
            # image_dir='dependencies/datasets/pha_kps/0106_V2/images/val',
            # annotation_file='dependencies/datasets/pha_kps/0106_V2/instances_val.json',
            image_dir='dependencies/datasets/pha_kps/0122_V5/images/val',
            annotation_file='dependencies/datasets/pha_kps/0122_V5/instances_val.json',
            # image_dir='dependencies/datasets/pha_object_detection/2025_02_06_01/images/images_crop',
            # annotation_file='dependencies/datasets/pha_object_detection/2025_02_06_01/9classes_crop_resize_hod.json',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=DATASET_CATEGORY_PHAKPTS)
        pha_kpts_val_cfg = dict(
            path=f'{settings.datasets_path}',
            split='images',
            # image_dir='dependencies/datasets/pha_kps/images_resize',
            # annotation_file='dependencies/datasets/pha_kps/annotaions/subset_instances_val2017_fix.json',
            # image_dir='dependencies/datasets/pha_kps_new/images',
            # annotation_file='dependencies/datasets/pha_kps_new/instances.json',
            # image_dir='dependencies/datasets/pha_kps/0106_V2/images/val',
            # annotation_file='dependencies/datasets/pha_kps/0106_V2/instances_val.json',
            image_dir='dependencies/datasets/pha_kps/0122_V5/images/val',
            annotation_file='dependencies/datasets/pha_kps/0122_V5/instances_val.json',            
            shuffle=False, # can be set to True as well, if needed
            # num_frames=min(settings.num_frames,5000),
            num_frames=settings.num_frames,
            name=DATASET_CATEGORY_PHAKPTS)
        dataset_cache[DATASET_CATEGORY_PHAKPTS]['calibration_dataset'] = COCOKeypoints(**pha_kpts_calib_cfg, download=False)
        dataset_cache[DATASET_CATEGORY_PHAKPTS]['input_dataset'] = COCOKeypoints(**pha_kpts_val_cfg, download=False)
    #
    if check_dataset_load(settings, DATASET_CATEGORY_WIDERFACE) and (DATASET_CATEGORY_WIDERFACE in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_WIDERFACE} variant:{DATASET_CATEGORY_WIDERFACE}"))
        widerface_det_calib_cfg = dict(
            path=f'{settings.datasets_path}/widerface',
            split='val',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=DATASET_CATEGORY_WIDERFACE)
        widerface_det_val_cfg = dict(
            path=f'{settings.datasets_path}/widerface',
            split='val',
            shuffle=False, # can be set to True as well, if needed
            num_frames=min(settings.num_frames,3226),
            name=DATASET_CATEGORY_WIDERFACE)
        dataset_cache[DATASET_CATEGORY_WIDERFACE]['calibration_dataset'] = WiderFaceDetection(**widerface_det_calib_cfg, download=download)
        dataset_cache[DATASET_CATEGORY_WIDERFACE]['input_dataset'] = WiderFaceDetection(**widerface_det_val_cfg, download=False)
    #
    if check_dataset_load(settings, DATASET_CATEGORY_COCOSEG21) and (DATASET_CATEGORY_COCOSEG21 in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_COCOSEG21} variant:{DATASET_CATEGORY_COCOSEG21}"))
        cocoseg21_calib_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=DATASET_CATEGORY_COCOSEG21)
        cocoseg21_val_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=min(settings.num_frames,5000),
            name=DATASET_CATEGORY_COCOSEG21)
        dataset_cache[DATASET_CATEGORY_COCOSEG21]['calibration_dataset'] = COCOSegmentation(**cocoseg21_calib_cfg, download=download)
        dataset_cache[DATASET_CATEGORY_COCOSEG21]['input_dataset'] = COCOSegmentation(**cocoseg21_val_cfg, download=False)
    #
    if check_dataset_load(settings, DATASET_CATEGORY_ADE20K) and (DATASET_CATEGORY_ADE20K in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_ADE20K} variant:{DATASET_CATEGORY_ADE20K}"))
        ade20k_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=DATASET_CATEGORY_ADE20K)
        ade20k_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=min(settings.num_frames, 2000),
            name=DATASET_CATEGORY_ADE20K)
        dataset_cache[DATASET_CATEGORY_ADE20K]['calibration_dataset'] = ADE20KSegmentation(**ade20k_seg_calib_cfg, download=download)
        dataset_cache[DATASET_CATEGORY_ADE20K]['input_dataset'] = ADE20KSegmentation(**ade20k_seg_val_cfg, download=False)
    #
    if check_dataset_load(settings, DATASET_CATEGORY_ADE20K32) and (DATASET_CATEGORY_ADE20K32 in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_ADE20K32} variant:{DATASET_CATEGORY_ADE20K32}"))
        ade20k_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=DATASET_CATEGORY_ADE20K32)
        ade20k_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=min(settings.num_frames, 2000),
            name=DATASET_CATEGORY_ADE20K32)
        dataset_cache[DATASET_CATEGORY_ADE20K32]['calibration_dataset'] = ADE20KSegmentation(**ade20k_seg_calib_cfg, num_classes=32, download=download)
        dataset_cache[DATASET_CATEGORY_ADE20K32]['input_dataset'] = ADE20KSegmentation(**ade20k_seg_val_cfg, num_classes=32, download=False)
    #
    if check_dataset_load(settings, DATASET_CATEGORY_VOC2012) and (DATASET_CATEGORY_VOC2012 in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_VOC2012} variant:{DATASET_CATEGORY_VOC2012}"))
        voc_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/VOCdevkit/VOC2012',
            split='val',
            shuffle=True,
            num_frames=settings.calibration_frames,
            name=DATASET_CATEGORY_VOC2012)
        voc_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/VOCdevkit/VOC2012',
            split='val',
            shuffle=True,
            num_frames=min(settings.num_frames, 1449),
            name=DATASET_CATEGORY_VOC2012)
        dataset_cache[DATASET_CATEGORY_VOC2012]['calibration_dataset'] = VOC2012Segmentation(**voc_seg_calib_cfg, download=download)
        dataset_cache[DATASET_CATEGORY_VOC2012]['input_dataset'] = VOC2012Segmentation(**voc_seg_val_cfg, download=False)
    #
    if check_dataset_load(settings, DATASET_CATEGORY_NYUDEPTHV2) and (DATASET_CATEGORY_NYUDEPTHV2 in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_NYUDEPTHV2} variant:{DATASET_CATEGORY_NYUDEPTHV2}"))
        filter_imgs = False
        nyudepthv2_calib_cfg = dict(
            path=f'{settings.datasets_path}/nyudepthv2',
            split='val',
            shuffle=True,
            num_frames=settings.calibration_frames,
            image_dir_path='/opt/code/edgeai-benchmark/dependencies/datasets/depth/valid/images',
            label_dir_path='/opt/code/edgeai-benchmark/dependencies/datasets/depth/valid/masks',
            name=DATASET_CATEGORY_NYUDEPTHV2)
        nyudepthv2_val_cfg = dict(
            path=f'{settings.datasets_path}/nyudepthv2',
            split='val',
            shuffle=False, #TODO: need to make COCODetection.evaluate() work with shuffle
            num_frames=min(settings.num_frames, 654),
            image_dir_path='/opt/code/edgeai-benchmark/dependencies/datasets/depth/valid/images',
            label_dir_path='/opt/code/edgeai-benchmark/dependencies/datasets/depth/valid/masks',
            name=DATASET_CATEGORY_NYUDEPTHV2)

        dataset_cache[DATASET_CATEGORY_NYUDEPTHV2]['calibration_dataset'] = NYUDepthV2(**nyudepthv2_calib_cfg, download=download)
        dataset_cache[DATASET_CATEGORY_NYUDEPTHV2]['input_dataset'] = NYUDepthV2(**nyudepthv2_val_cfg, download=False)
    #

    if check_dataset_load(settings, DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD) and (DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD} variant:{DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD}"))
        dataset_calib_cfg = dict(
            path=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd',
            split=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd/train_img_gt_pair.txt',
            num_classes=19,
            shuffle=True,
            num_frames=min(settings.calibration_frames,150),
            name=DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD
        )

        # dataset parameters for actual inference
        dataset_val_cfg = dict(
            path=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd',
            split=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd/val_img_gt_pair.txt',
            num_classes=19,
            shuffle=True,
            num_frames=min(settings.num_frames,49),
            name=DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD
        )

        dataset_cache[DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD]['calibration_dataset'] = RobokitSegmentation(**dataset_calib_cfg, download=True)
        dataset_cache[DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD]['input_dataset'] = RobokitSegmentation(**dataset_val_cfg, download=True)
    #

    if check_dataset_load(settings, DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD) and (DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD in dataset_list):
        print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD} variant:{DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD}"))
        dataset_calib_cfg = dict(
            path=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd',
            split=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd/train_img_gt_pair.txt',
            num_classes=19,
            shuffle=True,
            num_frames=min(settings.calibration_frames,150),
            name=DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD
        )

        # dataset parameters for actual inference
        dataset_val_cfg = dict(
            path=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd',
            split=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd/val_img_gt_pair.txt',
            num_classes=19,
            shuffle=True,
            num_frames=min(settings.num_frames,49),
            name=DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD
        )

        dataset_cache[DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD]['calibration_dataset'] = RobokitVisualLocalization(**dataset_calib_cfg, download=True)
        dataset_cache[DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD]['input_dataset'] = RobokitVisualLocalization(**dataset_val_cfg, download=True)
    #

    # the following are datasets cannot be downloaded automatically
    # put it under the condition of experimental_models
    if settings.experimental_models:
        if check_dataset_load(settings, DATASET_CATEGORY_CITYSCAPES) and (DATASET_CATEGORY_CITYSCAPES in dataset_list):
            print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_CITYSCAPES} variant:{DATASET_CATEGORY_CITYSCAPES}"))
            cityscapes_seg_calib_cfg = dict(
                path=f'{settings.datasets_path}/cityscapes',
                split='val',
                shuffle=True,
                num_frames=settings.calibration_frames,
                name=DATASET_CATEGORY_CITYSCAPES)
            cityscapes_seg_val_cfg = dict(
                path=f'{settings.datasets_path}/cityscapes',
                split='val',
                shuffle=True,
                num_frames=min(settings.num_frames,500),
                name=DATASET_CATEGORY_CITYSCAPES)
            dataset_cache[DATASET_CATEGORY_CITYSCAPES]['calibration_dataset'] = CityscapesSegmentation(**cityscapes_seg_calib_cfg, download=False)
            dataset_cache[DATASET_CATEGORY_CITYSCAPES]['input_dataset'] = CityscapesSegmentation(**cityscapes_seg_val_cfg, download=False)
        #
        if check_dataset_load(settings, DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS) and (DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS in dataset_list):
            print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS} variant:{DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS}"))
            dataset_calib_cfg = dict(
                path=f'{settings.datasets_path}/kitti_3dod/',
                split='training',
                pts_prefix='velodyne_reduced',
                num_classes=3,
                shuffle=False,
                num_frames=min(settings.calibration_frames, 3769),
                name=DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS)

            # dataset parameters for actual inference
            dataset_val_cfg = dict(
                path=f'{settings.datasets_path}/kitti_3dod/',
                split='training',
                pts_prefix='velodyne_reduced',
                num_classes=3,
                shuffle=False,
                num_frames=min(settings.num_frames, 3769),
                name=DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS)
            try:
                dataset_cache[DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS]['calibration_dataset'] = KittiLidar3D(**dataset_calib_cfg, download=False, read_anno=False)
                dataset_cache[DATASET_CATEGORY_KITTI_LIDAR_DET_3CLASS]['input_dataset'] = KittiLidar3D(**dataset_val_cfg, download=False, read_anno=True)
            except Exception as message:
                print(f'KittiLidar3D dataset loader could not be created: {message}')
            #
        #
        if check_dataset_load(settings, DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS) and (DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS in dataset_list):
            print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS} variant:{DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS}"))
            dataset_calib_cfg = dict(
                path=f'{settings.datasets_path}/kitti_3dod/',
                split='training',
                pts_prefix='velodyne_reduced',
                num_classes=1,
                shuffle=False,
                num_frames=min(settings.calibration_frames, 3769),
                name=DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS)

            # dataset parameters for actual inference
            dataset_val_cfg = dict(
                path=f'{settings.datasets_path}/kitti_3dod/',
                split='training',
                pts_prefix='velodyne_reduced',
                num_classes=1,
                shuffle=False,
                num_frames=min(settings.num_frames, 3769),
                name=DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS)
            try:
                dataset_cache[DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS]['calibration_dataset'] = KittiLidar3D(**dataset_calib_cfg, download=False, read_anno=False)
                dataset_cache[DATASET_CATEGORY_KITTI_LIDAR_DET_1CLASS]['input_dataset'] = KittiLidar3D(**dataset_val_cfg, download=False, read_anno=True)
            except Exception as message:
                print(f'KittiLidar3D dataset loader could not be created: {message}')
            #
        #

        if check_dataset_load(settings, DATASET_CATEGORY_KITTI_2015) and (DATASET_CATEGORY_KITTI_2015 in dataset_list):
            print(utils.log_color("\nINFO", f"loading dataset", f"category:{DATASET_CATEGORY_KITTI_2015} variant:{DATASET_CATEGORY_KITTI_2015}"))
            dataset_calib_cfg = dict(
                path=f'{settings.datasets_path}/kitti_2015/',
                split='training',                
                shuffle=False,
                max_disp=192,
                num_frames=min(settings.calibration_frames, 50))

            # dataset parameters for actual inference
            dataset_val_cfg = dict(
                path=f'{settings.datasets_path}/kitti_2015/',
                split='training',                
                shuffle=False,
                max_disp=192,
                num_frames=min(settings.num_frames, 50))
            try:
                dataset_cache['kitti_2015']['calibration_dataset'] = Kitti2015(**dataset_calib_cfg, download=False)
                dataset_cache['kitti_2015']['input_dataset'] = Kitti2015(**dataset_val_cfg, download=False)
            except Exception as message:
                print(f'Kitti 2015 dataset loader could not be created: {message}')


                
            #         
        #
    #
    return dataset_cache


def initialize_datasets(settings):
    dataset_cache = _initialize_datasets(settings)
    settings.dataset_cache = dataset_cache
    return True


def download_datasets(settings, download=True, dataset_list=None):
    # just creating the dataset classes with download=True will check of the dataset folders are present
    # if the dataset folders are missing, it will be downloaded and extracted
    # set download='always' to force re-download the datasets
    settings.dataset_cache = get_datasets(settings, download=download, dataset_list=dataset_list)
    return True


def _in_dataset_loading(settings, dataset_names):
    if settings.dataset_loading is None or settings.dataset_loading is True:
        return True
    elif settings.dataset_loading is False:
        return False
    #
    dataset_loading = utils.as_list(settings.dataset_loading)
    dataset_names = utils.as_list(dataset_names)
    for dataset_name in dataset_names:
        if dataset_name in dataset_loading:
            return True
        #
    #
    return False

def _in_dataset_selection(settings, dataset_names):
    if settings.dataset_selection is None or settings.dataset_selection is True:
        return True
    elif settings.dataset_selection is False:
        return False
    #
    dataset_selection = utils.as_list(settings.dataset_selection)
    dataset_names = utils.as_list(dataset_names)
    for dataset_name in dataset_names:
        if dataset_name in dataset_selection:
            return True
        #
    #
    return False

def check_dataset_load(settings, dataset_names):
    return _in_dataset_loading(settings, dataset_names) and _in_dataset_selection(settings, dataset_names)
