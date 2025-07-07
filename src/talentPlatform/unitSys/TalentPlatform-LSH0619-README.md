# 🤖 AI 드론을 활용한 산불 탐지 및 텔레그램 경보 시스템 (FireWatcher)

---

## 1. 프로젝트 개요

**FireWatcher**는 AI와 무인 이동체(드론) 기술을 융합하여 산불을 조기에 탐지하고, 신속하게 텔레그램으로 경보를 발송하는 시스템입니다. 드론이 촬영하는 실시간 영상 스트림을 YOLO(You Only Look Once) 객체 탐지 모델로 분석하여 연기와 화재를 식별하고, 지정된 텔레그램 채팅방으로 즉시 알림을 보냅니다. 이를 통해 산불 발생 시 골든타임을 확보하고 피해를 최소화하는 것을 목표로 합니다.

---

## 2. 주요 기능

-   **실시간 영상 분석**: 드론의 영상에서 `연기(smoke)`와 `불(fire)`을 실시간으로 탐지합니다.
-   **AI 기반 객체 탐지**: 최신 YOLO 모델을 활용하여 높은 정확도와 빠른 처리 속도로 객체를 식별합니다.
-   **텔레그램 알림**: 산불 징후 포착 시, 발생 시각, 탐지 종류(연기/불), 신뢰도(%) 정보를 포함한 메시지를 즉시 발송합니다.
-   **결과 저장**: 탐지 결과가 반영된 영상은 지정된 경로에 자동으로 저장되어 추후 분석에 활용할 수 있습니다.

---

## 3. 시스템 실행 환경

-   **플랫폼**: Google Colab
-   **프로그래밍 언어**: Python 3
-   **주요 라이브러리**:
    -   `ultralytics`
    -   `python-telegram-bot`
    -   `pytz`
    -   `gdown`
    -   `torch`, `torchvision`
    -   `opencv-python`

---

## 4. 실행 방법 (Google Colab)

본 프로젝트는 Google Colab 환경에서 간편하게 실행할 수 있도록 구성되었습니다.

#### **1단계: Colab에서 노트북 열기**

-   제공된 `UWC_2025_AI_&_무인이동체_퓨처_해커톤_소스_코드_FireWatcher.ipynb` 파일을 Google Colab에서 엽니다.

#### **2단계: 개발 환경 설정 (첫 번째 코드 셀 실행)**

-   첫 번째 코드 셀을 실행하여 Google Drive를 마운트하고 프로젝트에 필요한 디렉터리(`INPUT`, `OUTPUT` 등)를 생성합니다.
-   **Google Drive 접근 권한을 반드시 허용해 주세요.**

```python
# 구글 드라이브 연결 및 주요 경로 설정
import os
from google.colab import drive
# ... (이하 생략)
drive.mount('/content/drive')
# ... (이하 생략)
```

#### **3단계: 개발 환경 구축 (두 번째 코드 셀 실행)**

-   두 번째 코드 셀을 실행하면 다음 작업이 자동으로 수행됩니다
  -   산불 탐지를 위한 사전 학습된 YOLO 모델과 소스 코드를 다운로드하고 압축을 해제합니다.
  -   ultralytics, python-telegram-bot 등 프로젝트 실행에 필요한 라이브러리를 설치합니다.
  -   분석할 샘플 드론 영상(20250707_drone_fire.mp4)을 INPUT 폴더에 다운로드합니다.

```python
# 소스코드 및 라이브러리 다운로드
!gdown [https://drive.google.com/uc?id=1tRQdMF6knszG1P1rAeR7r6ZLkHdhlDKL](https://drive.google.com/uc?id=1tRQdMF6knszG1P1rAeR7r6ZLkHdhlDKL)
!unzip droneSmokeFireDetection.zip -d droneSmokeFireDetection

# 라이브러리 다운로드/설치
!pip install ultralytics
!pip install python-telegram-bot
!pip install pytz

# 샘플파일 다운로드
os.chdir(f"{globalVar['inpPath']}")
print(f"[CHECK] getcwd : {os.getcwd()}")
!gdown [https://drive.google.com/uc?id=18omItBxmzOzRBssdbd7EgCkJTiM5mkYz](https://drive.google.com/uc?id=18omItBxmzOzRBssdbd7EgCkJTiM5mkYz)
```
  
#### **4단계: 비즈니스 로직 실행**

-   주요 라이브러리 및 함수 선언
-   나머지 코드 셀들을 순서대로 실행하면 산불 탐지 및 경보 시스템이 작동합니다.
-   YOLO 모델이 로드되고, INPUT 폴더에 있는 샘플 영상을 분석하기 시작합니다.
-   영상에서 연기나 불이 탐지될 경우, 설정된 텔레그램으로 알림이 전송됩니다(10 프레임당 1회).
-   분석이 완료되면, 탐지 결과가 바운딩 박스로 표시된 영상이 OUTPUT 폴더에 저장됩니다.

```python
# 주요 라이브러리
import ultralytics
from ultralytics import YOLO
# ... (이하 생략)

# YOLO 모델 로드 및 예측 실행
model = YOLO(modeInfo)
results = model.predict(source=sourceInfo, conf=0.10, stream=True, save=True, project=os.path.dirname(outInfo), name=os.path.basename(outInfo))

cnt = 0
for r in results:
    # ... (탐지 및 알림 로직)
```

---

## 5. 디렉터리 구조

-   스크립트 실행 시 Google Drive 내 My Drive/COLAB/testSys-FireWatcher/ 경로에 다음과 같은 폴더가 생성됩니다.
```python
/testSys-FireWatcher/
├── droneSmokeFireDetection/  # 모델 및 소스코드
│   ├── datasets/
│   └── runs/
├── FIG/                      # 분석 결과 이미지 (현재 코드에서는 사용되지 않음)
├── INPUT/                    # 분석할 원본 영상 저장 경로
└── OUTPUT/                   # 탐지 결과 영상 저장 경로
```

---

