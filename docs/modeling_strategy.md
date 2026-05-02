# Modeling Strategy

## 1. Problem Formulation

본 프로젝트는 이미지 또는 동영상 입력에 대해 Real/Fake 확률을 출력하는 binary classification 문제로 정의된다. 동영상은 프레임 단위로 분해한 뒤 프레임별 확률을 영상 단위로 집계한다.

## 2. Data Strategy

공식 학습 데이터가 제공되지 않았기 때문에, 다양한 외부 딥페이크 데이터셋을 직접 구성했다. 특정 생성기나 특정 위조 방식에 과적합되지 않도록 GAN, Autoencoder, Diffusion, Face Reenactment, Inpainting, Fakechain 계열 데이터를 함께 사용했다.

## 3. Model Architecture

DINOv3 ViT-L/16 backbone은 freeze하고, LoRA adapter와 MLP classification head만 학습한다. 입력 token 중 REG1, REG2, patch token mean을 결합해 register-aware feature representation을 구성한다.

## 4. Training Strategy

- BCEWithLogitsLoss
- WeightedRandomSampler
- AdamW
- Gradient Accumulation
- CosineAnnealingLR
- AMP mixed precision
- Validation split with dataset and class balance consideration

## 5. Inference Strategy

이미지는 face crop 후 단일 추론을 수행한다. 동영상은 16개 프레임을 균등 추출하고, 프레임별 fake probability 중 상위 4개를 선택한 뒤 median을 최종 영상 점수로 사용한다.
