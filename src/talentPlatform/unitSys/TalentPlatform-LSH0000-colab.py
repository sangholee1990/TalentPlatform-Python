# ============================================
# 요구사항
# ============================================
# [요청] LSH0568. Python을 이용한 2024년 AI&무인이동체 퓨처 해커톤 (AI 북한군 탐지 및 경고 알림)

# ============================================
# 환경변수 설정
# ============================================
import os
from google.colab import drive
import glob
from IPython.display import HTML
from base64 import b64encode
import os

# 구글 드라이브 연결
drive.mount('/content/drive')

# 주요 설정
serviceName = 'LSH0568'
contextPath = f"/content/drive/MyDrive/COLAB/testSys-{serviceName}"

globalVar = {
    'ctxPath':  f"{contextPath}"
    , 'inpPath': f"{contextPath}/INPUT"
    , 'outPath': f"{contextPath}/OUTPUT"
    , 'figPath': f"{contextPath}/FIG"
}

for key, val in globalVar.items():
    if key.__contains__('Path'):
        os.makedirs(val, exist_ok=True)
        print(f"[CHECK] {key} : {val}")

# 작업 경로 설정
os.chdir(f"{globalVar['ctxPath']}")
print(f"[CHECK] getcwd : {os.getcwd()}")

# 옵션 설정
sysOpt = {

}

# 구글 드라이브 삭제
# ! rm -rf '/content/drive/MyDrive/COLAB/testSys-LSH0568'

# 구글 코랩 세션 무한 연결
# function PreventDisconnection(){
#     document.querySelector("colab-toolbar-button#connect").click()
#     console.log("클릭이 완료되었습니다.");
# }
# setInterval(PreventDisconnection, 60 * 10000)

# ============================================
# 시스템 환경 구축
# ============================================
# 소스코드 다운로드
# os.chdir(f"{globalVar['ctxPath']}")
# print(f"[CHECK] getcwd : {os.getcwd()}")
# ! git clone https://github.com/MuhammadMoinFaisal/YOLOv7-DeepSORT-Object-Tracking.git

# ============================================
# 비즈니스 로직
# ============================================
# 작업경로 설정
os.chdir(f"{globalVar['ctxPath']}/YOLOv7-DeepSORT-Object-Tracking")
os.getcwd()

# 테스트 파일 처리
srcInfo = f"{globalVar['ctxPath']}/YOLOv7-DeepSORT-Object-Tracking/TalentPlatform-LSH0568-deep_sort_tracking_id.py"
modelInfo = f"yolov7.pt"
inpInfo = f"{globalVar['inpPath']}/case_eGtahtswt_E.mp4"
outInfo = f"{globalVar['inpPath']}/result_20240710"

# 단위 테스트
# cmd = f"python {srcInfo} --weights {modelInfo} --source {inpInfo} --classes 0 --exist-ok --save_dir {os.path.basename(outInfo)}"
# print(f"[CHECK] cmd : {cmd}")
# ! {cmd}

# 통합 테스트
keyList = ["case_eGtahtswt_E.mp4", "case2_GVz6RZZmTlg.mp4", "case3_KHYEROW8RYA.mp4", "case4_Y2meta.mp4"]
for key in keyList:
    inpInfo = f"{globalVar['inpPath']}/{key}"
    cmd = f"python {srcInfo} --weights {modelInfo} --source {inpInfo} --classes 0 --save-txt --save-conf --exist-ok --save_dir {os.path.basename(outInfo)}"
    print(f"[CHECK] cmd : {cmd}")
    # ! {cmd}