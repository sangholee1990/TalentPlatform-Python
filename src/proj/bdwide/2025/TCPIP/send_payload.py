# cd /SYSTEMS/IOT/Roverdyn/PROJ_TCP_DB
# python send_payload.py
# python /SYSTEMS/IOT/Roverdyn/PROJ_TCP_DB/send_payload.py

import socket
import struct
import datetime 

# --- 설정 ---
HOST = '49.247.41.71'  # 대상 서버 IP 주소
# PORT = 9998            # 대상 서버 포트 번호
PORT = 9999            # 대상 서버 포트 번호
# 보낼 16진수 데이터 (공백 없이)
# EXPECTED_SOF = 0xFF  # 예시 SoF 값
# EXPECTED_SOF = 0xFc  # 예시 SoF 값
# EXPECTED_HEADER = b'\x00\x03\x07'
# PACKET_LENGTH = 11

# 정상
# hex_payload = 'FF00308207E84244574944452D30303333662D303561333737363739362D3839666634342D623762336563302D64333034303365343236323032352D30312D30312031303A30303A303041C1999A42B066664129999A4019999A6D6F76656D656E740000000000000000000000004090000042C8CCCD4039999A4290333341B2666642606666'

hex_payload = 'FF00308207E932423030333333343331353100000000000000000000000000000000000000000000000000000000000000000000000000323032352D30342D31382031343A34303A303041D6666641E4000040800000408000003000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
# hex_payload = 'FF000300'



# 이상
# hex_payload = 'FA00308207E84244574944452D30303333662D303561333737363739362D3839666634342D623762336563302D64333034303365343236323032352D30312D30312031303A30303A303041C1999A42B066664129999A4019999A6D6F76656D656E740000000000000000000000004090000042C8CCCD4039999A4290333341B2666642606666'



# (참고) 만약 "SoF 0x00 0x00 N" 헤더가 필요하다면 아래처럼 헤더를 추가하세요.

# SoF 536F46
# 0x00 00
# 0x00 00
# N len(hex_payload)
# header_hex = '536F4600004F' # N=0x4F (길이 79) 라고 가정
# header_hex = '%s%s%s%s'.format('536F46', '00', '00',  len(hex_payload) // 2)

# sof_hex = '536F46'
# null_byte_hex = '00'
# payload_len_dec = len(hex_payload) // 2
# # payload_len_dec = 78
# length_hex = format(payload_len_dec, '02X')

# header_hex = sof_hex + null_byte_hex + null_byte_hex + length_hex
# print(f"[CHECK] len(hex_payload) : {len(hex_payload)}")
# print(f"[CHECK] payload_len_dec : {payload_len_dec}")
# print(f"[CHECK] length_hex : {length_hex}")
# print(f"[CHECK] header_hex : {header_hex}")
# print(f"[CHECK] hex_payload : {hex_payload}")

# hex_to_send = header_hex + hex_payload
hex_to_send = hex_payload
# hex_to_send = hex_payload # 이 예제에서는 헤더 없이 페이로드만 보냅니다.


def decode_timestamp_packet(packet_data):
    """
    주어진 형식의 바이트 패킷을 디코딩하여 타임스탬프 정보를 반환합니다.
    성공 시 딕셔너리 반환, 실패 시 None 반환.
    """
    if not packet_data or len(packet_data) < PACKET_LENGTH:
        print("오류: 데이터 길이가 너무 짧습니다.")
        return None

    # 1. SoF 확인
    sof = packet_data[0]
    if sof != EXPECTED_SOF:
        print(f"오류: 잘못된 SoF입니다. (Expected: {EXPECTED_SOF:#04x}, Got: {sof:#04x})")
        return None

    # 2. 헤더 검증
    header = packet_data[1:4]
    if header != EXPECTED_HEADER:
        print(f"오류: 잘못된 헤더입니다. (Expected: {EXPECTED_HEADER.hex()}, Got: {header.hex()})")
        return None

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

try:
    # 16진수 문자열을 바이트 객체로 변환
    data_to_send = bytes.fromhex(hex_to_send)

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
        if received_data:
            # print(f"Received: {received_data.hex().upper()}")

            decoded_info = received_data.decode('ascii')
            # decoded_info = decode_timestamp_packet(received_data)
            print(f"Response Text: {received_data}")
            print(f"Response Text: {decoded_info}")
        else:
            print("No response or connection closed.")

except socket.error as e:
    print(f"Socket error: {e}")
except ValueError as e:
    print(f"Hex conversion error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    print("Closing connection.")
    # 'with' 구문이 자동으로 소켓을 닫아줍니다.
