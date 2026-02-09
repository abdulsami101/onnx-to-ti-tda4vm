#!/usr/bin/env python3
"""
Regression 데이터셋을 위한 샘플 이미지들을 생성하는 스크립트
"""

import cv2
import numpy as np
import os

def create_sample_images():
    """
    샘플 이미지들을 생성하여 regression 데이터셋에 추가
    """
    # 이미지 디렉토리 경로
    images_dir = "dependencies/datasets/regression/images"
    
    # 디렉토리가 없으면 생성
    os.makedirs(images_dir, exist_ok=True)
    
    # 샘플 이미지들 생성
    sample_images = [
        "image1.jpg",
        "image2.jpg", 
        "image3.jpg",
        "image4.jpg",
        "image5.jpg",
        "image6.jpg",
        "image7.jpg",
        "image8.jpg",
        "image9.jpg",
        "image10.jpg"
    ]
    
    print("샘플 이미지 생성 중...")
    
    for i, filename in enumerate(sample_images):
        # 랜덤한 색상의 이미지 생성 (224x224, 3채널)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # 파일 경로
        filepath = os.path.join(images_dir, filename)
        
        # 이미지 저장
        cv2.imwrite(filepath, img)
        print(f"생성됨: {filepath}")
    
    print(f"\n{len(sample_images)}개의 샘플 이미지가 {images_dir}에 생성되었습니다.")

if __name__ == "__main__":
    create_sample_images()
