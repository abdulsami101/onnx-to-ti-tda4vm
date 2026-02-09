#!/usr/bin/env python3
"""
Regression 모델을 ONNX로 변환하는 스크립트
"""

import torch
import torch.nn as nn
from torchvision import models
import os

def create_regression_model():
    """
    Regression 모델 생성
    """
    # RegNet 모델 로드
    model = models.get_model(
        name="regnet_x_800mf", 
        weights=models.RegNet_X_800MF_Weights.DEFAULT
    )
    
    # 마지막 fully connected 레이어를 regression용으로 변경 (2차원 출력)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    return model

def convert_to_onnx(model, output_path, input_shape=(1, 3, 224, 224)):
    """
    PyTorch 모델을 ONNX로 변환
    
    Args:
        model: PyTorch 모델
        output_path: ONNX 파일 저장 경로
        input_shape: 입력 텐서 형태 (batch_size, channels, height, width)
    """
    # 모델을 evaluation 모드로 설정
    model.eval()
    
    # 더미 입력 생성
    dummy_input = torch.randn(input_shape)
    
    # ONNX로 변환
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"모델이 {output_path}에 저장되었습니다.")
    print(f"입력 형태: {input_shape}")
    print(f"출력 형태: (batch_size, 2)")

def main():
    """
    메인 함수
    """
    # 모델 생성
    print("Regression 모델 생성 중...")
    model = create_regression_model()
    
    # 모델 저장 디렉토리 생성
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # ONNX 파일 경로
    onnx_path = os.path.join(models_dir, "regnet_x_800mf.onnx")
    
    # ONNX로 변환
    print("ONNX로 변환 중...")
    convert_to_onnx(model, onnx_path)
    
    print("변환 완료!")

if __name__ == "__main__":
    main()
