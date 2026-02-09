#!/bin/bash

# Regression 모델을 위한 benchmark 실행 스크립트

# 설정
TARGET_SOC="TDA4VM"
SETTINGS_FILE="settings_regression.yaml"

echo "=== Regression Model Benchmark 시작 ==="

# 1단계: 샘플 이미지 생성
echo "1단계: 샘플 이미지 생성..."
python3 create_sample_images.py

# 2단계: 모델 변환 (ONNX)
echo "2단계: PyTorch 모델을 ONNX로 변환..."
python3 convert_regression_model.py

# 3단계: 모델 import 및 calibration (중요: inference는 실행하지 않음)
echo "3단계: 모델 import 및 calibration..."
python3 ./scripts/benchmark_modelzoo.py ${SETTINGS_FILE} --target_device ${TARGET_SOC} --run_inference False

# 4단계: 추론 실행 (중요: import는 실행하지 않음)
echo "4단계: 추론 실행..."
python3 ./scripts/benchmark_modelzoo.py ${SETTINGS_FILE} --target_device ${TARGET_SOC} --run_import False

echo "=== Regression Model Benchmark 완료 ==="
