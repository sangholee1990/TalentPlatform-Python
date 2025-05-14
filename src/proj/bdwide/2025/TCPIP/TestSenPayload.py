# cd /SYSTEMS/IOT/Roverdyn/PROJ_TCP_DB
# python send_payload.py
# python /SYSTEMS/IOT/Roverdyn/PROJ_TCP_DB/send_payload.py

import socket
import struct
import datetime 

def decode_timestamp_packet(packet_data):
    """
    주어진 형식의 바이트 패킷을 디코딩하여 타임스탬프 정보를 반환합니다.
    성공 시 딕셔너리 반환, 실패 시 None 반환.
    """
    # if not packet_data or len(packet_data) < PACKET_LENGTH:
    #     print("오류: 데이터 길이가 너무 짧습니다.")
    #     return None

    # 1. SoF 확인
    sof = packet_data[0]
    # if sof != EXPECTED_SOF:
    #     print(f"오류: 잘못된 SoF입니다. (Expected: {EXPECTED_SOF:#04x}, Got: {sof:#04x})")
    #     return None

    # 2. 헤더 검증
    header = packet_data[1:4]
    # if header != EXPECTED_HEADER:
    #     print(f"오류: 잘못된 헤더입니다. (Expected: {EXPECTED_HEADER.hex()}, Got: {header.hex()})")
    #     return None

    # 3. 데이터 추출 및 디코딩 (struct 모듈 사용)
    try:
        # '>HBBBBB' : Big-endian unsigned short(Year), 5 * unsigned char(M,D,H,M,S)
        # 데이터 부분(Year부터)은 인덱스 4부터 시작
        year, month, day, hour, minute, second = struct.unpack('>HBBBBB', packet_data[4:11])

        # 간단한 유효성 검사 (필요에 따라 더 추가)
        if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
             print("경고: 날짜/시간 값이 유효 범위를 벗어났습니다.")
             # 오류로 처리할 수도 있음: return None

        return {
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "minute": minute,
            "second": second
        }

    except struct.error as e:
        print(f"오류: 데이터 언패킹 실패 - {e}")
        return None
    except Exception as e:
        print(f"오류: 디코딩 중 예외 발생 - {e}")
        return None

if __name__ == '__main__':

    HOST = '49.247.41.71'
    PORT = 9999

    # 정상
    # hex_payload = 'FF000300'
    # hex_payload = 'FF00308207E932423030333333343331353100000000000000000000000000000000000000000000000000000000000000000000000000323032352D30342D31382031343A34303A303041D6666641E4000040800000408000003000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
    # hex_payload = 'FF00141907e900000004323032352d30312d33312032303a30303a3030'
    hex_payload = 'FF00151907E900000004323032352D30382D30312030303A30303A3030'

    try:
        # 16진수 문자열을 바이트 객체로 변환
        data_to_send = bytes.fromhex(hex_payload.upper())

        # 소켓 생성 및 연결
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"Connecting to {HOST}:{PORT}...")
            s.connect((HOST, PORT))
            print(f"Connected.")

            # 데이터 전송
            print(f"Sending {len(data_to_send)} bytes...")
            s.sendall(data_to_send)
            print(f"Data sent: {data_to_send.hex().upper()}")

            # (선택) 서버 응답 받기
            print("Waiting for response...")
            received_data = s.recv(1024) # 최대 1024 바이트 읽기

            if not received_data:
                raise Exception("No response or connection closed.")

            if hex_payload in ['FF000300']:
                decoded_info = decode_timestamp_packet(received_data)
            else:
                decoded_info = received_data.decode('ascii')
            print(f"[CHECK] decoded_info : {decoded_info}")

    except Exception as e:
        print(f"Exception : {e}")
    finally:
        print("Closing connection.")
