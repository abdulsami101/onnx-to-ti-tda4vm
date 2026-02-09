import os
import shutil
import cv2
import pickle
import json
import random
from colorama import Fore
from .. import utils
from .image_cls import *

class RegressionGazeDataset(ImageClassification):
    """
    Regression을 위한 데이터셋 클래스
    파일명과 ground truth 값을 가진 JSON 파일을 사용
    PHA_CLS 구조와 동일하게 구현
    """
    
    def __init__(self, *args, image_path, annotations_file, download=False, num_frames=None, name='regression', **kwargs):
        # super().__init__()

        if download:
            raise Exception("No download support.")

        assert annotations_file.endswith('json')
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        self.imgs = list()
        self.labels = list()  # regression targets (labels로 이름 변경하여 호환성 유지)

        # JSON 파일에서 이미지 경로와 target 값들을 로드
        for k, v in data.items():
            this_image_path = os.path.join(image_path, k)
            assert os.path.exists(this_image_path), f"Image not found: {this_image_path}"
            self.imgs.append(this_image_path)
            self.labels.append(v)  # target 값 (리스트 또는 단일 값)
        
        shuffle = kwargs.get('shuffle', False)
        if shuffle:
            seed = 314  # to reproduce calibration process
            random.seed(seed)
            random.shuffle(self.imgs)
            random.seed(seed)
            random.shuffle(self.labels)

        self.num_frames = len(self.imgs) if num_frames is None else num_frames
        self.kwargs = kwargs

        # 데이터셋 정보 설정 (PHA_CLS와 동일한 구조)
        info = dict(description='Regression Dataset', url='', version='1.0',
                    year='2024', contributor='', date_created='')
        
        # Regression의 경우 categories는 필요 없지만, 호환성을 위해 빈 리스트로 설정
        categories = []
        
        self.dataset_store = dict(info=info, categories=categories)
        self.kwargs['dataset_info'] = self.get_dataset_info()

        super().initialize()

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx, **kwargs):
        with_label = kwargs.get('with_label', False)
        if with_label:
            return self.imgs[idx], self.labels[idx]
        else:
            return self.imgs[idx]
    
    def evaluate(self, predictions, **kwargs):
        """
        Regression 평가 함수
        MSE (Mean Squared Error) 계산
        """
        metric_tracker = utils.AverageMeter(name='mse')
        num_frames = min(self.num_frames, len(predictions))
        
        for n in range(num_frames):
            gt_label = self.labels[n]  # labels로 이름 통일
            if isinstance(predictions[n], list):
                pred = predictions[n][0]
            else:
                pred = predictions[n]
            
            # MSE 계산
            mse = self.regression_accuracy(pred, gt_label, **kwargs)
            metric_tracker.update(mse)
        
        return {metric_tracker.name: metric_tracker.avg}
    
    def regression_accuracy(self, prediction, target, **kwargs):
        """
        Regression의 경우 MSE를 반환
        PHA_CLS의 classification_accuracy와 동일한 인터페이스 유지
        """
        import numpy as np
        if isinstance(prediction, dict):
            prediction = prediction.get('preds', prediction)
        pred = np.array(prediction).squeeze()
        # pred = np.array(prediction)
        target = np.array(target)
        mse = np.mean((pred - target) ** 2)
        return mse
    
    def get_notice(self):
        notice = f'{Fore.YELLOW}' \
                 f'\nRegression Dataset'  \
                 f'{Fore.RESET}\n'
        return notice

    def get_dataset_info(self):
        if self.dataset_store is None:
            return None
        #
        # return only info and categories for now as the whole thing could be quite large.
        dataset_store = dict()
        for key in ('info', 'categories'):
            if key in self.dataset_store.keys():
                dataset_store.update({key: self.dataset_store[key]})
            #
        #
        print("num_class:" ,len(self.dataset_store['categories']))
        dataset_store.update(dict(color_map=self.get_color_map(num_classes=len(self.dataset_store['categories']))))        
        return dataset_store
