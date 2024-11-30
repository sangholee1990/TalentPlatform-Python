import csv
import json
import os
import paho.mqtt.client as mqtt

import pandas as pd

import traceback
import sys

data_path = '/HDD/DATA/BEE-IOT/CSV/'
# data_path = '/home/dms01user01/iot_server/mqtt/'
file_name = 'data.csv'

CSV_FILE = data_path + file_name
# CSV_FILE = file_name

# subscriber callback
def on_message(client, userdata, message):
        try:
                received_message = str(message.payload.decode("utf-8"))
                print("message received ",received_message)
                print("message topic= ", message.topic)
                print("message qos=", message.qos)
                print("message retain flag= ", message.retain)
                received_data = json.loads(received_message)
                new_row = pd.DataFrame.from_records([received_data])
                
                if not os.path.exists(CSV_FILE):
                        new_df = new_row
                        new_df.to_csv(CSV_FILE, index=False)
                                
                else:                               
                        df = pd.read_csv(CSV_FILE)
                        if not new_row.isin(df.to_dict(orient='list')).all(axis=None):
                                df = pd.concat([df, new_row], ignore_index=True)
                        df.to_csv(CSV_FILE, index=False)
                                        
                print(f"Data saved to {CSV_FILE}: {received_data}")

        except Exception as e:
                print(f"Failed to process the message: {e}")
                exc_type, exc_value, exc_traceback = sys.exc_info()  # 예외 정보 가져오기
                traceback_details = traceback.extract_tb(exc_traceback)  # 추적 정보 추출
                for trace in traceback_details:
                        print(f"File {trace.filename}, line {trace.lineno}, in {trace.name}")

broker_address = "0.0.0.0"
client1 = mqtt.Client("client1")
client1.connect(broker_address)
client1.subscribe("/topic/test1")
client1.on_message = on_message
client1.loop_forever()