# DINOv3 ViT-LoRA Deepfake Detection

DINOv3 ViT-L/16 백본과 LoRA 기반 파인튜닝을 활용해 이미지 및 동영상 콘텐츠의 딥페이크 여부를 판별하는 딥페이크 탐지 프로젝트입니다.

본 저장소는 대회 제출용 코드에서 대용량 모델 가중치와 학습·평가 데이터를 제외하고, 학습 코드, 추론 코드, 모델 구조, 데이터셋 로더, 설정 파일, 실행 환경을 정리한 버전입니다.

## Competition Result

| 항목 | 내용 |
|---|---|
| 대회명 | HAI(하이)! - Hecto AI Challenge : 2025 하반기 헥토 채용 AI 경진대회 |
| 주제 | 딥페이크 탐지 AI 모델 개발 |
| 문제 유형 | 이미지·동영상 기반 Real/Fake Binary Classification |
| 최종 성과 | 대상, 1등 |
| 대회 링크 | https://dacon.io/competitions/official/236628/overview/description |

## 1. 프로젝트 개요

본 프로젝트의 목표는 다양한 이미지 또는 동영상 입력에 대해 해당 콘텐츠가 실제 이미지인지, 딥페이크로 생성·조작된 콘텐츠인지 판별하는 것입니다.

대회에서는 별도의 공식 학습 데이터가 제공되지 않았기 때문에, 학습 데이터 구축, 전처리, 증강, 모델 학습, 추론 전략을 모두 직접 설계해야 했습니다. 이에 따라 특정 생성 방식에만 과적합되지 않도록 GAN, Autoencoder, Diffusion, Face Reenactment, Inpainting, Fakechain 등 다양한 위조 체계를 포함하는 학습 구성을 사용했습니다.

## 2. 핵심 접근 방식

| 구분 | 전략 |
|---|---|
| Backbone | DINOv3 ViT-L/16 pretrained backbone |
| Fine-tuning | Backbone freeze + LoRA adapter tuning |
| Classification Head | Register-aware head: REG1, REG2, patch token statistics 결합 |
| Input Resolution | 448 × 448 resize |
| Loss | BCEWithLogitsLoss |
| Sampling | WeightedRandomSampler |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Efficiency | AMP mixed precision, gradient accumulation |
| Inference | Face detection crop, 16-frame sampling, Top-4 Median aggregation |

## 3. 모델 구조

<pre>
Input Image / Video Frame
        |
        v
Resize 448 x 448
        |
        v
DINOv3 ViT-L/16 Backbone
        |
        +-- CLS Token
        +-- Register Tokens
        +-- Patch Tokens
        |
        v
Register-aware Feature Aggregation
        - REG1
        - REG2
        - Patch Token Mean
        |
        v
MLP Head
        |
        v
Sigmoid Probability: P(Fake)
</pre>

기존 ViT 기반 분류는 주로 CLS token 하나에 의존합니다. 하지만 딥페이크 탐지에서는 배경, 압축, 조명, 저정보 영역에서 발생하는 spurious signal이 CLS 표현에 섞일 수 있습니다. 본 프로젝트는 register token과 patch token 통계를 함께 사용해 얼굴 질감, 국소 통계 불일치, 배경 노이즈를 더 안정적으로 분리하도록 설계했습니다.

## 4. 데이터 구성 전략

학습 데이터는 특정 생성기 또는 특정 조작 방식에만 과적합되지 않도록 다음 유형을 포함했습니다.

| 그룹 | 포함 데이터 |
|---|---|
| Real | FFHQ, Pexels, KoDF real frames 등 |
| GAN / Autoencoder 계열 | HiDF, FaceForensics++, KoDF 등 |
| Diffusion 계열 | DiffusionFace, Stable Diffusion 기반 합성 이미지 |
| 복합 위조 | Face Reenactment, Inpainting, Fakechain 유형 |
| 추가 검증용 | MFFI 등 차세대 생성 모델 기반 데이터 |

데이터 라이선스와 원본 출처는 각 데이터셋의 정책을 따르며, 본 저장소에는 대용량 원본 데이터가 포함되어 있지 않습니다.

## 5. 데이터 전처리 및 증강

### 5.1 학습 전처리

학습 단계에서는 얼굴 정렬, landmark 기반 정규화, 과도한 crop을 사용하지 않고, 이미지를 448 × 448로 resize하는 최소 전처리를 적용했습니다. 이는 데이터셋별 crop 방식 차이로 인해 새로운 편향이 발생하는 것을 줄이고, 얼굴 주변 맥락 정보까지 보존하기 위한 선택입니다.

### 5.2 Augmentation

실제 서비스 환경에서 발생할 수 있는 화질 저하와 분포 이동에 대응하기 위해 다음 증강을 사용했습니다.

| 유형 | 목적 |
|---|---|
| HorizontalFlip | 좌우 방향 변화에 대한 기본 강건성 |
| Affine | 위치·스케일·회전 변화에 대한 강건성 |
| RandomBrightnessContrast | 조명 변화 대응 |
| HueSaturationValue | 색조·채도 변화 대응 |
| ImageCompression | SNS·메신저 업로드 압축 노이즈 대응 |
| MotionBlur | 영상 프레임 흔들림 대응 |
| GaussianBlur | 저화질·초점 흐림 대응 |

## 6. 학습 전략

본 프로젝트는 DINOv3의 사전학습 표현을 최대한 보존하면서, 딥페이크 탐지에 필요한 변별 정보만 효율적으로 주입하는 방향으로 설계했습니다.

- DINOv3 ViT-L/16 backbone freeze
- Attention 및 MLP Linear layer 중심 LoRA 적용
- 전체 파라미터 중 일부 adapter와 MLP head만 학습
- BCEWithLogitsLoss 기반 binary classification
- WeightedRandomSampler로 class imbalance 완화
- AdamW와 gradient accumulation으로 안정적인 업데이트
- AMP mixed precision으로 학습 메모리와 속도 최적화
- validation split은 생성 방식과 데이터셋 편중을 줄이도록 구성

## 7. 추론 전략

추론 단계에서는 이미지와 동영상을 모두 처리할 수 있도록 구성했습니다.

### 7.1 이미지 추론

이미지는 InsightFace 기반 얼굴 검출 후, 얼굴 bbox를 scale ratio 1.4로 확장하여 crop합니다. 얼굴이 검출되지 않는 경우에는 center crop fallback을 적용합니다.

### 7.2 동영상 추론

동영상은 전체 프레임을 모두 처리하지 않고 16개 프레임을 균등 샘플링합니다. 각 프레임의 fake probability를 계산한 뒤, 상위 4개 확률에 대해 median aggregation을 적용합니다.

<pre>
Video
  -> Uniform 16-frame sampling
  -> Face crop per frame
  -> DINOv3-LoRA detector
  -> Frame-level fake probabilities
  -> Top-4 selection
  -> Median aggregation
  -> Final fake probability
</pre>

Top-K Median 방식은 전체 평균보다 outlier frame의 영향을 줄이면서, 딥페이크 단서가 강하게 나타나는 프레임을 더 적극적으로 반영할 수 있습니다.

## 8. 저장소 구조

<pre>
dinov3-vit-lora-deepfake-detection/
├─ config/
│  └─ config.yaml
├─ env/
│  ├─ Dockerfile
│  └─ requirements.txt
├─ src/
│  ├─ dataset.py
│  ├─ models.py
│  └─ utils.py
├─ docs/
│  ├─ modeling_strategy.md
│  └─ competition_summary.md
├─ train.py
├─ inference.py
├─ README.md
├─ requirements.txt
├─ .dockerignore
└─ .gitignore
</pre>

## 9. 실행 방법

### 9.1 환경 설치

<pre>
python -m pip install --upgrade pip
pip install -r requirements.txt
</pre>

### 9.2 학습 데이터 배치

대용량 학습 데이터는 GitHub에 포함하지 않습니다. 학습 재현 시 아래 구조로 직접 배치합니다.

<pre>
train_data/
└─ train_data_zips/
   └─ train_full_data.zip
</pre>

### 9.3 학습 실행

<pre>
python train.py --config config/config.yaml
</pre>

학습 결과는 기본적으로 `model/` 하위에 저장됩니다.

<pre>
model/
├─ checkpoints/
├─ model.pt
└─ inference.pth
</pre>

### 9.4 추론 실행

추론에는 DINOv3 backbone, LoRA/head 가중치, InsightFace model 파일이 필요합니다. 해당 파일은 용량 문제로 GitHub에 포함하지 않습니다.

<pre>
python inference.py
</pre>

추론 결과는 기본적으로 다음 경로에 저장됩니다.

<pre>
test_data/submission.csv
</pre>

## 10. GitHub 업로드 제외 항목

다음 항목은 용량과 라이선스 문제로 저장소에 포함하지 않습니다.

- `model/`
- `train_data/`
- `test_data/`
- `*.pt`, `*.pth`, `*.safetensors`, `*.bin`
- 원본 학습 데이터 zip
- 대회 평가 데이터

## 11. 설계 의의

이 프로젝트는 단순히 backbone을 교체한 딥페이크 탐지 모델이 아니라, 다음 관점에서 실전형 탐지 파이프라인을 구성한 프로젝트입니다.

- 공식 학습 데이터가 없는 상황에서 외부 데이터셋을 직접 구성
- GAN, Autoencoder, Diffusion, 복합 위조 파이프라인까지 고려한 일반화 중심 데이터 설계
- DINOv3의 사전학습 표현을 보존하면서 LoRA로 효율적인 task adaptation 수행
- register-aware feature aggregation 적용
- 이미지와 동영상을 모두 처리하는 통합 추론 파이프라인 구성
- 동영상 추론에서 Top-4 Median aggregation으로 오판 프레임의 영향을 완화
