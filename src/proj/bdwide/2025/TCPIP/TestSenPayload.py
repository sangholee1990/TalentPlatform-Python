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
        print(f"오류: 데이터 언패킹 실패 : {e}")
        return None
    except Exception as e:
        print(f"오류: 디코딩 중 예외 발생 : {e}")
        return None

if __name__ == '__main__':

    HOST = '49.247.41.71'
    PORT = 9999
    # PORT = 9998

    # 운영
    # hex_payload = 'FF00000401020304'
    # hex_payload = 'FF00308207E933343030333233343331353100000000000000000000000000000000000000000000000000000000000000000000000000323032352D30352D31392031333A30303A303041F4CCCD41E6666641A8000041A80000300000000000000000000000000000000000000040C000004401C00043CC8000000000000000000000000000'
    # hex_payload = 'FF000300'
    # hex_payload = 'FF00141907E900000004323032352D30352D31392031333A30303A3030'
    hex_payload = 'FF00151907E900000004323032352D30352D30312030303A30303A3030'

    # 정상
    # hex_payload = 'FF00000401020304'
    # hex_payload = 'FF000300'
    # hex_payload = 'FF00308207E932423030333333343331353100000000000000000000000000000000000000000000000000000000000000000000000000323032352D30342D31382031343A34303A303041D6666641E4000040800000408000003000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
    # hex_payload = 'FF00141907e900000004323032352d30312d33312032303a30303a3030'
    # hex_payload = 'FF00151907E900000004323032352D30382D30312030303A30303A3030'
    # hex_payload = 'FF00084607E7323032332D30362D31352030333A30303A30304244574944452D30303333662D303561333737363739362D3839666634342D623762336563302D64333034303365343236'

    # 이상
    # hex_payload = 'FA000300'
    # hex_payload = 'FA000300 FF00308207E932423030333333343331353100000000000000000000000000000000000000000000000000000000000000000000000000323032352D30342D31382031343A34303A303041D6666641E4000040800000408000003000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'

    hex_payloads_to_send = [
        'FF000300',
        "FF00000401020304",
        "FF00308207E933343030333233343331353100000000000000000000000000000000000000000000000000000000000000000000000000323032352D30352D31392031333A30303A303041F4CCCD41E6666641A8000041A80000300000000000000000000000000000000000000040C000004401C00043CC8000000000000000000000000000",
    ]


    try:
        # # 16진수 문자열을 바이트 객체로 변환
        # data_to_send = bytes.fromhex(hex_payload.upper())
        #
        # # 소켓 생성 및 연결
        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #     print(f"Connecting to {HOST}:{PORT}...")
        #     s.connect((HOST, PORT))
        #     print(f"Connected.")
        #
        #     # 데이터 전송
        #     print(f"Sending {len(data_to_send)} bytes...")
        #     s.sendall(data_to_send)
        #     print(f"Data sent: {data_to_send.hex().upper()}")
        #
        #     # (선택) 서버 응답 받기
        #     print("Waiting for response...")
        #     received_data = s.recv(1024) # 최대 1024 바이트 읽기
        #
        #     if not received_data:
        #         raise Exception("No response or connection closed.")
        #
        #     if hex_payload in ['FF000300']:
        #         decoded_info = decode_timestamp_packet(received_data)
        #     elif hex_payload in ['FF00000401020304']:
        #         decoded_info = received_data
        #     else:
        #         decoded_info = received_data.decode('ascii')
        #     print(f"[CHECK] decoded_info : {decoded_info}")

        print(f"Connecting to {HOST}:{PORT}...")
        # 소켓 생성 및 연결 (루프 바깥에서 한 번만 수행)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.settimeout(5.0)  # 응답 대기 시간 5초 (선택 사항)
            print(f"Connected to server.")
            print("-" * 30)

            # N개의 메시지를 순차적으로 전송 및 수신
            for i, current_hex_payload in enumerate(hex_payloads_to_send):
                print(
                    f"Processing message {i + 1}/{len(hex_payloads_to_send)}: Payload literal = '{current_hex_payload}'")
                try:
                    # 16진수 문자열을 바이트 객체로 변환
                    data_to_send = bytes.fromhex(current_hex_payload.upper())

                    # 데이터 전송
                    print(f"  Sending {len(data_to_send)} bytes: {data_to_send.hex().upper()}")
                    s.sendall(data_to_send)
                    print(f"  Data sent.")

                    # (선택) 서버 응답 받기
                    print(f"  Waiting for response...")
                    received_data = s.recv(1024)  # 최대 1024 바이트 읽기

                    if not received_data:
                        print(f"  No response from server or connection closed for message {i + 1}.")
                        # 연결이 끊어졌다고 판단되면 루프를 중단할 수 있습니다.
                        break

                    print(f"  Received ({len(received_data)} bytes): {received_data.hex().upper()}")

                    decoded_info = ""
                    # 보낸 current_hex_payload를 기준으로 응답을 어떻게 디코딩할지 결정
                    if current_hex_payload.upper() == 'FF000300':  # GET_SYSTEM_TIME 요청에 대한 응답
                        decoded_info = decode_timestamp_packet(received_data)
                    elif current_hex_payload.upper() == 'FF00000401020304':  # MSG_ID 0x0000 요청에 대한 응답
                        # 사용자의 원본 코드에서는 이 경우 received_data를 그대로 사용했습니다.
                        # C++ 서버가 이 메시지에 대해 IP 주소 문자열을 보낸다면 decode('ascii')가 적절합니다.
                        # decoded_info = received_data # 원본 로직 (바이트 객체 유지)
                        try:
                            decoded_info = received_data.decode('ascii')  # 일반적으로 IP 주소는 ASCII
                            print(f"  Decoded as ASCII for 0x0000 cmd.")
                        except UnicodeDecodeError:
                            decoded_info = f"Non-ASCII response for 0x0000 cmd: {received_data.hex().upper()}"

                    else:  # 기타 다른 메시지에 대한 응답 (기본적으로 ASCII로 시도)
                        try:
                            decoded_info = received_data.decode('ascii')
                            print(f"  Decoded as ASCII for other cmd.")
                        except UnicodeDecodeError:
                            decoded_info = f"Non-ASCII response: {received_data.hex().upper()}"

                    print(f"  [CHECK] Decoded Info for message {i + 1}: {decoded_info}")

                except socket.timeout:
                    print(f"  Socket timeout for message {i + 1}. Server may not have responded in time.")
                except ConnectionResetError:
                    print(f"  Connection reset by server during message {i + 1}.")
                    break  # 연결이 완전히 끊어졌으므로 루프 중단
                except Exception as e_inner:
                    print(f"  Exception during message {i + 1} ({current_hex_payload}): {e_inner}")

                print("-" * 30)
                # if i < len(hex_payloads_to_send) - 1:
                #     time.sleep(0.2)  # 다음 메시지 전송 전 짧은 대기 (선택 사항)

    except Exception as e:
        print(f"Exception : {e}")
    finally:
        print("Closing connection.")
