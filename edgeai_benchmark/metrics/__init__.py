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

import numpy as np
from .. import utils

class MetricMSE():
    """
    Mean Squared Error (MSE) 계산을 위한 메트릭 클래스
    """
    def __init__(self, name='mse', label_offset_pred=None):
        self.name = name
        self.metric_tracker = utils.AverageMeter(name=name)
        self.label_offset_pred = label_offset_pred  # 호환성을 위한 속성 추가
    
    def __call__(self, predictions, targets, **kwargs):
        """
        MSE 계산
        
        Args:
            predictions: 모델 예측값 (numpy array)
            targets: ground truth 값 (numpy array)
            
        Returns:
            dict: MSE 값이 포함된 딕셔너리
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # MSE 계산
        mse = np.mean((predictions - targets) ** 2)
        self.metric_tracker.update(mse)
        
        return {self.name: self.metric_tracker.avg}
    
    def reset(self):
        """메트릭 트래커 초기화"""
        self.metric_tracker.reset()
