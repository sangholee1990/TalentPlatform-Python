import asyncio
import websockets
import time
import wave
import os
from pydub import AudioSegment

# WAV 파일 정보
SAMPLE_RATE = 44100  # PCM 데이터의 샘플링 레이트 (ESP32 측에서 동일해야 함)
CHANNELS = 1  # 모노(1) 또는 스테레오(2)
SAMPLE_WIDTH = 2  # 16비트 PCM 데이터 (2바이트)
SAMPLE_INTERVAL = 1 / SAMPLE_RATE  # 샘플레이트 기준 간격 (초)

SAVE_INTERVAL = 10   # 저장 간격
PING_INTERVAL = 5   # ping 간격
EXPECTED_DATA_SIZE = SAMPLE_RATE * SAMPLE_WIDTH * SAVE_INTERVAL  # 예상 데이터 크기
audio_data_buffer = bytearray()

# 무음 데이터를 정의 (16비트 샘플 기준)
def generate_silence(duration_seconds):
    """지정한 시간 동안 무음 데이터를 생성 (초 단위)"""
    silence_samples = SAMPLE_RATE * SAMPLE_WIDTH * duration_seconds
    return bytearray([0] * silence_samples)

async def save_audio_to_wav():
    """주기적으로 버퍼에 있는 PCM 데이터를 WAV 파일로 저장"""
    global audio_data_buffer
    while True:
        await asyncio.sleep(SAVE_INTERVAL)
        if audio_data_buffer:
            # 데이터 크기 확인
            received_data_size = len(audio_data_buffer)
            if received_data_size != EXPECTED_DATA_SIZE:
                print(f"Warning: Expected {EXPECTED_DATA_SIZE} bytes, but received {received_data_size} bytes.")
                # 부족한 부분을 무음으로 채우기
                missing_size = EXPECTED_DATA_SIZE - received_data_size
                audio_data_buffer.extend(generate_silence(missing_size // SAMPLE_WIDTH))
            
            timestamp = int(time.time())
            filename = f'audio_{timestamp}.wav'
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(SAMPLE_WIDTH)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_data_buffer)
            print(f"Audio saved to {filename}")
            
            sound = AudioSegment.from_wav(f'audio_{timestamp}.wav')
            amplified_sound = sound + 20  # 볼륨을 20dB 증폭
            amplified_sound.export(f'audio_{timestamp}_amplified.wav', format="wav")
            
            audio_data_buffer = bytearray()  # 버퍼 초기화

async def handle_connection(websocket, path):
    """WebSocket 연결을 처리하고, 데이터를 버퍼에 추가"""
    global audio_data_buffer
    last_receive_time = time.time()  # 첫 수신 시간
    print("Client connected")
    try:
        async for message in websocket:            
            current_time = time.time()
            elapsed_time = current_time - last_receive_time
            expected_samples = int(elapsed_time / SAMPLE_INTERVAL)  # 지난 시간에 해당하는 샘플 수 계산


            # 정상 수신 간격보다 시간이 초과되었을 경우, 무음으로 채우기
            if elapsed_time > SAMPLE_INTERVAL:
                missed_samples = expected_samples - (len(message) // SAMPLE_WIDTH)
                if missed_samples > 0:
                    print(f"Filling {missed_samples} samples of silence for {elapsed_time:.6f} seconds gap.")
                    audio_data_buffer.extend(generate_silence(missed_samples))
                    
            else:
                # PCM 바이너리 데이터를 버퍼에 추가
                audio_data_buffer.extend(message)
    except websockets.ConnectionClosed as e:
        print(f"Connection closed: {e}")
        
async def handle_ping(websocket, path):
    """Handle incoming pings"""
    try:
        # Override to detect when a ping is received
        pong_waiter = await websocket.ping()
        print("Ping received from client!")
        await pong_waiter  # Responds with pong automatically

    except websockets.ConnectionClosed:
        print("Connection closed after ping")

async def main():
    # WebSocket 서버 시작
    server = await websockets.serve(handle_connection, "0.0.0.0", 8888, ping_interval=PING_INTERVAL, ping_timeout=None)
    print("WebSocket Server started on port 8888")

    # 파일 저장 작업 시작
    await asyncio.gather(server.wait_closed(), save_audio_to_wav())

if __name__ == "__main__":
    asyncio.run(main())
