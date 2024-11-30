import asyncio
import websockets
import time
from pydub import AudioSegment
import io

SAMPLE_RATE = 44100  # PCM 데이터의 샘플링 레이트
CHANNELS = 1  # 모노
SAMPLE_WIDTH = 2  # 16비트 PCM 데이터 (2바이트)
SAVE_INTERVAL = 10  # 저장 간격
PING_INTERVAL = 10  # ping 전송 간격
audio_data_buffer = bytearray()

async def save_audio_to_mp3():
    """주기적으로 버퍼에 있는 PCM 데이터를 MP3 파일로 저장"""
    global audio_data_buffer
    while True:
        await asyncio.sleep(SAVE_INTERVAL)
        if audio_data_buffer:
            timestamp = int(time.time())
            filename = f'audio_{timestamp}.mp3'
            
            # PCM 데이터를 pydub AudioSegment로 변환
            audio_segment = AudioSegment(
                data=bytes(audio_data_buffer),  # PCM 데이터
                sample_width=SAMPLE_WIDTH,
                frame_rate=SAMPLE_RATE,
                channels=CHANNELS
            )
            
            # MP3 파일로 저장
            audio_segment.export(filename, format="mp3")
            print(f"Audio saved to {filename}")
            audio_data_buffer = bytearray()  # 버퍼 초기화
            
            sound = AudioSegment.from_mp3(f'audio_{timestamp}.mp3')
            amplified_sound = sound + 20  # 볼륨을 db만큼 증폭
            amplified_sound.export(f'audio_{timestamp}_amplified.mp3', format="mp3")

async def handle_connection(websocket, path):
    """WebSocket 연결을 처리하고, 데이터를 버퍼에 추가"""
    global audio_data_buffer
    print("Client connected")
    try:
        async for message in websocket:
            # PCM 바이너리 데이터를 버퍼에 추가
            audio_data_buffer.extend(message)
    except websockets.ConnectionClosed as e:
        print(f"Connection closed: {e}")

async def main():
    # WebSocket 서버 시작
    server = await websockets.serve(handle_connection, "0.0.0.0", 8888, ping_interval=PING_INTERVAL, ping_timeout=None)
    print("WebSocket Server started on port 8888")

    # 파일 저장 작업 시작
    await asyncio.gather(server.wait_closed(), save_audio_to_mp3())

if __name__ == "__main__":
    asyncio.run(main())
