import os
import shutil
import cv2
import pickle
import json
import random
from colorama import Fore
from .. import utils
from .image_cls import *

class RegressionFacelmDataset(ImageClassification):
    """
    Regression을 위한 데이터셋 클래스
    파일명과 ground truth 값을 가진 JSON 파일을 사용
    xy와 v를 분리하여 저장
    """
    
    def __init__(self, *args, image_path, annotations_file, download=False, num_frames=None, name='regression', **kwargs):
        if download:
            raise Exception("No download support.")

        assert annotations_file.endswith('json')
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        self.imgs = list()
        self.labels = list()   # 전체 (xy+v) ground truth
        self.labels_xy = list()
        self.labels_v  = list()

        # JSON 파일에서 이미지 경로와 target 값들을 로드
        for k, v in data.items():
            this_image_path = os.path.join(image_path, k)
            assert os.path.exists(this_image_path), f"Image not found: {this_image_path}"
            self.imgs.append(this_image_path)

            v = list(v)  # 보장: list 형태
            self.labels.append(v)

            # xy와 v 분리
            xy = [val for i, val in enumerate(v) if i % 3 != 2]
            vv = [val for i, val in enumerate(v) if i % 3 == 2]
            self.labels_xy.append(xy)
            self.labels_v.append(vv)
        
        shuffle = kwargs.get('shuffle', False)
        if shuffle:
            seed = 314  # to reproduce calibration process
            random.seed(seed)
            random.shuffle(self.imgs)
            random.seed(seed)
            random.shuffle(self.labels)
            random.seed(seed)
            random.shuffle(self.labels_xy)
            random.seed(seed)
            random.shuffle(self.labels_v)

        self.num_frames = len(self.imgs) if num_frames is None else num_frames
        self.kwargs = kwargs

        # 데이터셋 정보 설정 (PHA_CLS와 동일한 구조)
        info = dict(description='Regression FaceLandMark Dataset ', url='', version='1.0',
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
            return self.imgs[idx], {
                'xy': self.labels_xy[idx],
                'v': self.labels_v[idx]
            }
        else:
            return self.imgs[idx]
    
    def evaluate(self, predictions, **kwargs):
        """
        Regression 평가 함수
        MSE (Mean Squared Error) 계산 (xy 값만 사용)
        """
        metric_tracker = utils.AverageMeter(name='mse_xy')
        num_frames = min(self.num_frames, len(predictions))
        
        for n in range(num_frames):
            gt_label_xy = self.labels_xy[n]

            if isinstance(predictions[n], dict):
                pred_xy = predictions[n].get('preds', predictions[n])  # preds는 xy 값
            else:
                pred_xy = predictions[n]
            
            mse = self.regression_accuracy(pred_xy, gt_label_xy, **kwargs)
            metric_tracker.update(mse)
        
        return {metric_tracker.name: metric_tracker.avg}
    
    def regression_accuracy(self, prediction, target, **kwargs):
        """
        Regression의 경우 MSE를 반환 (xy 값만 사용)
        """
        import numpy as np
        if isinstance(prediction, dict):
            prediction = prediction.get('preds', prediction)
        pred = np.array(prediction).squeeze()
        target = np.array(target)
        mse = np.mean((pred - target) ** 2)
        return mse
    
    def get_notice(self):
        notice = f'{Fore.YELLOW}' \
                 f'\nRegression Dataset (Facelm: xy, v 분리)'  \
                 f'{Fore.RESET}\n'
        return notice

    def get_dataset_info(self):
        if self.dataset_store is None:
            return None
        #
        dataset_store = dict()
        for key in ('info', 'categories'):
            if key in self.dataset_store.keys():
                dataset_store.update({key: self.dataset_store[key]})
        #
        dataset_store.update(dict(color_map=self.get_color_map(num_classes=len(self.dataset_store['categories']))))        
        return dataset_store
