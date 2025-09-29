import logging
import sys
import requests
import pandas as pd
from datetime import timedelta
import mysql.connector as mysql
import configparser

# 옵션 설정
sysOpt = {
    'sysPath': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
}

config = configparser.ConfigParser()
config.read(sysOpt['sysPath'], encoding='utf-8')

# -------------------------
# 로깅 설정
# -------------------------
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("error.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# -------------------------
# DB 설정
# -------------------------
DB_CONFIG = {
    "host": config.get('mysql-iwin-dms01user01-DMS03', 'host'),
    "user": config.get('mysql-iwin-dms01user01-DMS03', 'user'),
    "password": config.get('mysql-iwin-dms01user01-DMS03', 'pwd'),
    "database":  config.get('mysql-iwin-dms01user01-DMS03', 'dbName')
}

# -------------------------
# API 설정
# -------------------------
BASE_URL =  config.get('ecowitt-device1', 'BASE_URL')
APPLICATION_KEY = config.get('ecowitt-device1', 'APPLICATION_KEY')
API_KEY = config.get('ecowitt-device1', 'API_KEY')

# -------------------------
# 기본값 설정 (2025.09.18 기준 장비 1개로 인한 기본값 설정)
# -------------------------
DEFAULT_MAC = config.get('ecowitt-device1', 'DEFAULT_MAC')
DEFAULT_DEVICE_ID = config.get('ecowitt-device1', 'DEFAULT_DEVICE_ID')

# -------------------------
# 데이터 필터 및 매핑
# -------------------------
required_data = [
    ("outdoor", "temperature"), ("outdoor", "humidity"),
    ("indoor", "temperature"), ("indoor", "humidity"),
    ("solar_and_uvi", "solar"), ("solar_and_uvi", "uvi"),
    ("rainfall_piezo", "rain_rate"), ("rainfall_piezo", "daily"),
    ("rainfall_piezo", "state"), ("wind", "wind_speed"),
    ("wind", "wind_direction"), ("pressure", "relative"),
    ("pressure", "absolute"), ("co2_aqi_combo", "co2"),
    ("pm25_aqi_combo", "real_time_aqi"), ("pm10_aqi_combo", "real_time_aqi"),
    ("t_rh_aqi_combo", "temperature"), ("t_rh_aqi_combo", "humidity"),
    ("temp_and_humidity_ch1", "temperature"), ("temp_and_humidity_ch2", "temperature"),
    ("temp_and_humidity_ch3", "temperature"), ("temp_and_humidity_ch1", "humidity"),
    ("temp_and_humidity_ch2", "humidity"), ("temp_and_humidity_ch3", "humidity"),
    ("temp_ch1", "temperature"), ("battery", "haptic_array_battery"),
    ("battery", "haptic_array_capacitor"), ("battery", "aqi_combo_sensor"),
    ("battery", "temp_humidity_sensor_ch1"), ("battery", "temp_humidity_sensor_ch2"),
    ("battery", "temp_humidity_sensor_ch3"), ("battery", "temperature_sensor_ch1")
]

column_mapping = {
    ("outdoor", "temperature"): "outdoor_temp",
    ("outdoor", "humidity"): "outdoor_hmdty",
    ("indoor", "temperature"): "indoor_temp",
    ("indoor", "humidity"): "indoor_hmdty",
    ("solar_and_uvi", "solar"): "solar",
    ("solar_and_uvi", "uvi"): "uvi",
    ("rainfall_piezo", "rain_rate"): "rain_rate",
    ("rainfall_piezo", "daily"): "rain_daily",
    ("rainfall_piezo", "state"): "rain_state",
    ("wind", "wind_speed"): "wind_speed",
    ("wind", "wind_direction"): "wind_direction",
    ("pressure", "relative"): "pressure_relative",
    ("pressure", "absolute"): "pressure_absolute",
    ("co2_aqi_combo", "co2"): "co2",
    ("pm25_aqi_combo", "real_time_aqi"): "pm25",
    ("pm10_aqi_combo", "real_time_aqi"): "pm10",
    ("t_rh_aqi_combo", "temperature"): "aqi_temp",
    ("t_rh_aqi_combo", "humidity"): "aqi_hmdty",
    ("temp_and_humidity_ch1", "temperature"): "temp1",
    ("temp_and_humidity_ch2", "temperature"): "temp2",
    ("temp_and_humidity_ch3", "temperature"): "temp3",
    ("temp_and_humidity_ch1", "humidity"): "hmdty1",
    ("temp_and_humidity_ch2", "humidity"): "hmdty2",
    ("temp_and_humidity_ch3", "humidity"): "hmdty3",
    ("temp_ch1", "temperature"): "temp_ch1",
    ("battery", "haptic_array_battery"): "haptic_array_battery",
    ("battery", "haptic_array_capacitor"): "haptic_array_capacitor",
    ("battery", "aqi_combo_sensor"): "aqi_battey",
    ("battery", "temp_humidity_sensor_ch1"): "temp1_battery",
    ("battery", "temp_humidity_sensor_ch2"): "temp2_battery",
    ("battery", "temp_humidity_sensor_ch3"): "temp3_battery",
    ("battery", "temperature_sensor_ch1"): "temp_ch1_battery"
}

default_values = {col: -999 for col in column_mapping.values()}
default_values["rain_state"] = "0"

# -------------------------
# 함수 정의
# -------------------------
def flatten_sensor_data(data_dict):
    records = []
    for category, sensors in data_dict.items():
        for sensor, details in sensors.items():
            if isinstance(details, dict) and "value" in details:
                records.append({
                    "category": category,
                    "sensor": sensor,
                    "value": details["value"],
                    "unit": details.get("unit", ""),
                    "timestamp": details.get("time", "")
                })
    return pd.DataFrame(records)

def insert_ecowitt_data(df, device_id):
    conn, cursor = None, None
    try:
        conn = mysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
            INSERT INTO TB_ECOWITT_DATA (
                tm, outdoor_temp, outdoor_hmdty, indoor_temp, indoor_hmdty, solar, uvi,
                rain_rate, rain_daily, wind_speed, wind_direction, pressure_relative,
                pressure_absolute, co2, pm25, pm10, aqi_temp, aqi_hmdty, temp1,
                temp2, temp3, hmdty1, hmdty2, hmdty3, temp_ch1, aqi_battey,
                temp1_battery, temp2_battery, temp3_battery, temp_ch1_battery,
                rain_state, device_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """

        records_to_insert = []
        for _, row in df.iterrows():
            records_to_insert.append(tuple([
                row["tm"], row["outdoor_temp"], row["outdoor_hmdty"], row["indoor_temp"],
                row["indoor_hmdty"], row["solar"], row["uvi"], row["rain_rate"],
                row["rain_daily"], row["wind_speed"], row["wind_direction"],
                row["pressure_relative"], row["pressure_absolute"], row["co2"],
                row["pm25"], row["pm10"], row["aqi_temp"], row["aqi_hmdty"],
                row["temp1"], row["temp2"], row["temp3"], row["hmdty1"], row["hmdty2"],
                row["hmdty3"], row["temp_ch1"], row["aqi_battey"], row["temp1_battery"],
                row["temp2_battery"], row["temp3_battery"], row["temp_ch1_battery"],
                row["rain_state"], device_id
            ]))

        cursor.executemany(query, records_to_insert)
        conn.commit()

        # 2025.09.19
        pd.set_option('display.max_columns', None)
        print(f"[CHECK] device_id : {device_id} / final_df : {final_df}")
    except mysql.Error as err:
        if conn: conn.rollback()
        logging.error(f"데이터베이스 오류: {err}")
    except Exception as e:
        if conn: conn.rollback()
        logging.error(f"일반 오류: {e}")
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

# -------------------------
# 메인 실행
# -------------------------
if __name__ == "__main__":
    try:
        mac_address = DEFAULT_MAC
        device_id = DEFAULT_DEVICE_ID
        
        if len(sys.argv) > 1:
            mac_address = sys.argv[1]
        
        if len(sys.argv) > 2:
            device_id = int(sys.argv[2])

        PARAMS = {
            "application_key": APPLICATION_KEY,
            "api_key": API_KEY,
            "mac": mac_address,
            "temp_unitid": 1,
            "pressure_unitid": 3,
            "wind_speed_unitid": 6
        }

        response = requests.get(BASE_URL, params=PARAMS)
        if response.status_code == 200:
            data = response.json().get("data", {})
            df = flatten_sensor_data(data)
    
            # 수정된 부분: timestamp 컬럼을 먼저 숫자형으로 변환
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["datetime_kst"] = pd.to_datetime(df["timestamp"], unit="s", utc=True) + timedelta(hours=9)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

            df_filtered = df[df[["category", "sensor"]].apply(tuple, axis=1).isin(required_data)].copy()
            df_filtered["column"] = df_filtered.apply(lambda row: column_mapping.get((row["category"], row["sensor"])), axis=1)

            pivoted_df = df_filtered.pivot_table(
                index=["timestamp", "datetime_kst"], columns="column", values="value"
            ).reset_index()

            pivoted_df["datetime_kst"] = pd.to_datetime(pivoted_df["datetime_kst"]).dt.strftime("%Y-%m-%d %H:%M:%S")

            final_df = pivoted_df.reindex(columns=["datetime_kst"] + list(default_values.keys()))
            final_df = final_df.rename(columns={"datetime_kst": "tm"}).fillna(value=default_values)
    
            insert_ecowitt_data(final_df, device_id)
        else:
            logging.error(f"API 호출 실패 : {response.status_code}")

    except Exception as e:
        logging.error(f"스크립트 실행 중 오류 발생 : {e}")
