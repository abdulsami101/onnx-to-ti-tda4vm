
import os
import random
import pickle
import numpy as np

from .. import utils
from .dataset_base import *

class ImageRegression(DatasetBase):
    def __init__(self, download=False, dest_dir=None, num_frames=None, name=None, **kwargs):
        super().__init__(num_frames=num_frames, name=name, **kwargs)
        self.force_download = True if download == 'always' else False
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'path and split must be provided in kwargs'
        assert 'num_classes' in self.kwargs, f'num_classes must be provided while creating {self.__class__.__name__}'
        assert name is not None, 'Please provide a name for this dataset'

        path = self.kwargs['path']
        split_file = self.kwargs['split']

        #
        assert os.path.exists(path) and os.path.isdir(path), \
            utils.log_color('\nERROR', 'dataset path is empty', path)

        # create list of images and classes
        self.imgs = [os.path.join(self.path, 'images', i) for i in os.listdir(os.path.join(self.path, 'images'))]
        label_path = os.path.join(self.path, 'labels.pkl')
        with open(label_path, 'rb') as f:
            self.labels = pickle.load(f)
            
        self.num_frames = self.kwargs['num_frames'] = self.kwargs.get('num_frames',len(self.imgs))
        shuffle = self.kwargs.get('shuffle', False)
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.imgs)
        #

    def download(self, path, split_file):
        return None

    @staticmethod
    def get_name(img_path):
        return img_path.split('/')[-1]
    
    def __getitem__(self, idx, **kwargs):
        with_label = kwargs.get('with_label', False)
        img_path = self.imgs[idx]
        name = self.get_name(img_path)
        if with_label:
            label = self.labels[name]
            return img_path, label
        else:
            return img_path
        #

    def __len__(self):
        return self.num_frames

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, predictions, **kwargs):
        metric_tracker = utils.AverageMeter(name='mse')
        num_frames = min(self.num_frames, len(predictions))
        for n in range(num_frames):
            name = self.get_name(self.imgs[n])
            gt_label = self.labels[name]
            # accuracy = self.classification_accuracy(predictions[n], gt_label, **kwargs)
            mse = self.mse(predictions[n], gt_label)
            metric_tracker.update(mse)
        #
        return {metric_tracker.name:metric_tracker.avg}

    # def classification_accuracy(self, prediction, target, label_offset_pred=0, label_offset_gt=0,
    #                             multiplier=100.0, **kwargs):
    #     prediction = prediction + label_offset_pred
    #     target = target + label_offset_gt
    #     accuracy = 1.0 if (prediction == target) else 0.0
    #     accuracy = accuracy * multiplier
    #     return accuracy

    def mse(self, prediction, target):
        if prediction.shape!=target.shape:
            prediction = prediction.squeeze()
            target = target.squeeze()
        assert prediction.shape==target.shape
        return np.sqrt(np.sum((prediction-target)**2))
