import asyncio
import websockets
import cv2
import numpy as np
import time
from datetime import datetime

# Set the video parameters
frame_width = 800
frame_height = 600
fps = 30  # Frames per second
output_format = "avi"  # "mp4" or "avi"
segment_duration = 10  # Duration of each video segment in seconds
PING_INTERVAL = 10   # ping 간격
frame_interval = 1 / fps  # Time interval between frames

# Function to generate filename based on the current time
def generate_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"output_{timestamp}.{output_format}"

# Function to create a new video writer object
def create_video_writer():
    if output_format == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_format == "avi":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        raise ValueError("Unsupported video format. Use 'mp4' or 'avi'.")
    return cv2.VideoWriter(generate_filename(), fourcc, fps, (frame_width, frame_height))

# Initialize the video writer for the first segment
out = create_video_writer()
start_time = time.time()
last_frame = None  # Store the last received frame

async def handle_frame(websocket, path):
    global out, start_time, last_frame
    print("Connection established, waiting for frames...")    
    frames_received = 0  # Keep track of frames received
    
    while True:
        try:
            # Receive frame from WebSocket (binary data)
            frame_data = await websocket.recv()
                        
            if(isinstance(frame_data, bytes)):
                # Convert the received bytes into a numpy array
                nparr = np.frombuffer(frame_data, np.uint8)

                # Decode the image (assuming JPEG frames)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                # Update the last received frame
                last_frame = frame
            else:
                print("Received an empty frame, using last known frame.")

            # Resize the frame if necessary (optional)
            resized_frame = cv2.resize(last_frame, (frame_width, frame_height))

            # Write the frame into the video file
            out.write(resized_frame)
            
            # Increment frame counter
            frames_received += 1

            # Check if it's time to create a new video file
            elapsed_time = time.time() - start_time
            
            if elapsed_time >= segment_duration:
            # if frames_received >= total_frames:
                # Close the current video file
                out.release()
                
                print("release file.")
                print(f"start time : {time.strftime('%Y-%m-%d %I:%M:%S %p', time.localtime(start_time))}")
                print(f"release time : {time.strftime('%Y-%m-%d %I:%M:%S %p', time.localtime(time.time()))}")

                # Create a new video file
                out = create_video_writer()
                
                # Reset the frame counter
                frames_received = 0
                start_time = time.time()
        except websockets.ConnectionClosed:
            print("Connection closed")
            out.release()
            break
        # Sleep for frame interval to match desired FPS
        await asyncio.sleep(frame_interval)

async def main():
    # Start WebSocket server
    async with websockets.serve(handle_frame, "0.0.0.0", 8765, ping_interval=PING_INTERVAL, ping_timeout=None):
        print("Server listening on ws://0.0.0.0:8765")
        await asyncio.Future()  # Keep running

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Server stopped")

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()