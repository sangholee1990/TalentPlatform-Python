from http import HTTPStatus
from flask import Flask, send_file, request, jsonify
import json

app = Flask(__name__)

# 펌웨어 파일 경로 설정
SENSOR_FIRMWARE_PATH = "c3.bin"
CAM_FIRMWARE_PATH = "cam.bin"

# 기본 경로에서 펌웨어 파일을 제공하는 엔드포인트
@app.route('/sensor/ota', methods=['GET'])
def sensor_ota():
    # 사용자 에이전트 확인 (선택사항)
    user_agent = request.headers.get('User-Agent')
    print(f"OTA 요청 from: {user_agent}")

    try:
        # 펌웨어 파일을 바이너리 형식으로 반환
        return send_file(SENSOR_FIRMWARE_PATH, as_attachment=True, download_name='c3.bin')
    except Exception as e:
        return f"Error: {str(e)}", 500
    
@app.route('/sensor/version', methods=['GET'])
def sernsor_version():    
    # 사용자 에이전트 확인 (선택사항)
    user_agent = request.headers.get('User-Agent')
    print(f"version 확인 요청 from: {user_agent}")
    
    try:
        # 펌웨어 파일을 바이너리 형식으로 반환
        with open('version.json', 'r') as f:
            version_data = json.load(f)
            return version_data
    except Exception as e:
        return f"Error: {str(e)}", 500
    

# 기본 경로에서 펌웨어 파일을 제공하는 엔드포인트
@app.route('/cam/ota', methods=['GET'])
def cam_ota():
    # 사용자 에이전트 확인 (선택사항)
    user_agent = request.headers.get('User-Agent')
    print(f"OTA 요청 from: {user_agent}")

    try:
        # 펌웨어 파일을 바이너리 형식으로 반환
        return send_file(CAM_FIRMWARE_PATH, as_attachment=True, download_name='cam.bin')
    except Exception as e:
        return f"Error: {str(e)}", 500
    
@app.route('/cam/version', methods=['GET'])
def cam_version():    
    # 사용자 에이전트 확인 (선택사항)
    user_agent = request.headers.get('User-Agent')
    print(f"version 확인 요청 from: {user_agent}")
    
    try:
        # 펌웨어 파일을 바이너리 형식으로 반환
        with open('version_cam.json', 'r') as f:
            version_data = json.load(f)
            return version_data
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    # 서버 시작
    app.run(host='0.0.0.0', port=3000, debug=True)