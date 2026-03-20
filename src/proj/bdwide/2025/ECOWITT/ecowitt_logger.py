import time
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


# File path
LOG_FILE = 'ecowitt_log.txt'
CSV_FILE = 'ecowitt_api.csv'

# Logging function
def log(message):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{now_str}] {message}\n")

# Flatten DataFrame
def flatten_sensor_data(data_dict):
    records = []
    for category, sensors in data_dict.items():
        for sensor, details in sensors.items():
            if isinstance(details, dict) and 'value' in details:
                record = {
                    "datetime_kst": datetime.now(ZoneInfo("Asia/Seoul")),
                    "category": category,
                    "sensor": sensor,
                    "value": details["value"],
                    "timestamp": details.get("time", "")
                }
                records.append(record)
    return pd.DataFrame(records)

# Save CSV File
def append_to_csv(new_df, filename=CSV_FILE):
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    combined_df.to_csv(filename, index=False)

# Ecowitt API configuration
base_url = "https://api.ecowitt.net/api/v3/device/real_time"
params = {
    "application_key": "B2A840FDA6C5475BBD2E58ACCB4CA3DA",
    "api_key": "e6fcf427-d379-4299-8ef0-fd397af1dd98",
    "mac": "1C:69:20:E3:A9:97",
    "temp_unitid": 1,
    "pressure_unitid": 3,
    "wind_speed_unitid": 6
}

while True:
    start_time = datetime.now()

    if 0 <= start_time.second <= 20:
        log(f"{start_time.second} seconds: Waiting 20 seconds to avoid stale data")
        time.sleep(20)
        start_time = datetime.now()

    try:
        # Call the API and collect data
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json().get("data", {})
            df = flatten_sensor_data(data)
            append_to_csv(df)
        else:
            log(f"API error code: {response.status_code}")
    except Exception as e:
        log(f"Exception occurred: {e}")

    # Sleep until the next scheduled collection time
    next_run = start_time + timedelta(minutes=1)
    sleep_time = (next_run - datetime.now()).total_seconds()
    if sleep_time > 0:
        time.sleep(sleep_time)
