# ============================================
# 요구사항
# ============================================
# Python을 이용한 에코위트 장비 수집체계

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# /HDD/SYSTEMS/LIB/anaconda3/envs/mysql_env/bin/python /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/ECOWITT/ecowitt_collector.py
# */5 * * * * cd  /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/ECOWITT && /HDD/SYSTEMS/LIB/anaconda3/envs/mysql_env/bin/python /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/ECOWITT/ecowitt_collector.py

# ============================================
# 라이브러리
# ============================================
import logging
import sys
import requests
import pandas as pd
from datetime import timedelta
import mysql.connector as mysql
import configparser
import logging
import logging.handlers
import os
import platform

# 옵션 설정
sysOpt = {
    # 'sysPath': '/HDD/SYSTEMS/PROG/PYTHON/IDE/resources/config/system.cfg',
    'sysPath': '/HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/ECOWITT/system.cfg',
}

config = configparser.ConfigParser()
config.read(sysOpt['sysPath'], encoding='utf-8')


# ============================================
# 유틸리티 함수
# ============================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
    )

    os.makedirs(os.path.dirname(saveLogFile), exist_ok=True)

    log = logging.getLogger(prjName)

    if len(log.handlers) > 0:
        return log

    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')
    streamHandler = logging.StreamHandler()
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=saveLogFile, when='midnight', interval=1, backupCount=30, encoding='utf-8')

    streamHandler.setFormatter(format)
    fileHandler.setFormatter(format)

    log.addHandler(streamHandler)
    log.addHandler(fileHandler)

    log.setLevel(level=logging.INFO)

    return log

# -------------------------
# 로깅 설정
# -------------------------
# logging.basicConfig(
#     level=logging.ERROR,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler("error.log"),
#         logging.StreamHandler(sys.stdout)
#     ]
# )

env = 'dev'
prjName = 'ecowitt_collector'
# ctxPath = os.getcwd()
ctxPath = f"/HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2025/ECOWITT"

log = initLog(env, ctxPath, prjName)

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
# 기본값 설정 (2025.11.04 기준 장비 추가로 인해 config값 변경하고 뒤로 이동)
# -------------------------
# DEFAULT_MAC = config.get('ecowitt-device1', 'MAC')
# DEFAULT_DEVICE_ID = config.get('ecowitt-device1', 'DEVICE_ID')

# -------------------------
# 데이터 필터 및 매핑
# -------------------------
# 산숲팜 장비용
sansup_required_data = [
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

sansup_column_mapping = {
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

sansup_default_values = {col: -999 for col in sansup_column_mapping.values()}
sansup_default_values["rain_state"] = "0"

# 가평 이수근 실외 장비용(2025.11.04)
external_required_data = [
    ("outdoor", "temperature"), ("outdoor", "humidity"),
    ("solar_and_uvi", "solar"), ("solar_and_uvi", "uvi"),
    ("rainfall_piezo", "rain_rate"), ("rainfall_piezo", "daily"),
    ("rainfall_piezo", "state"), ("wind", "wind_speed"),
    ("wind", "wind_direction"), ("co2_aqi_combo", "co2"),
    ("pm25_aqi_combo", "real_time_aqi"), ("pm10_aqi_combo", "real_time_aqi"),
    ("t_rh_aqi_combo", "temperature"), ("t_rh_aqi_combo", "humidity"),
    ("temp_and_humidity_ch1", "temperature"), ("temp_and_humidity_ch1", "humidity"),
    ("battery", "haptic_array_battery"), ("battery", "haptic_array_capacitor"), 
    ("battery", "aqi_combo_sensor"), ("battery", "temp_humidity_sensor_ch1"),
    ("battery", "console")
]

external_column_mapping = {
    ("outdoor", "temperature"): "outdoor_temp",
    ("outdoor", "humidity"): "outdoor_hmdty",
    ("solar_and_uvi", "solar"): "solar",
    ("solar_and_uvi", "uvi"): "uvi",
    ("rainfall_piezo", "rain_rate"): "rain_rate",
    ("rainfall_piezo", "daily"): "rain_daily",
    ("rainfall_piezo", "state"): "rain_state",
    ("wind", "wind_speed"): "wind_speed",
    ("wind", "wind_direction"): "wind_direction",
    ("co2_aqi_combo", "co2"): "co2",
    ("pm25_aqi_combo", "real_time_aqi"): "pm25",
    ("pm10_aqi_combo", "real_time_aqi"): "pm10",
    ("t_rh_aqi_combo", "temperature"): "aqi_temp",
    ("t_rh_aqi_combo", "humidity"): "aqi_hmdty",
    ("temp_and_humidity_ch1", "temperature"): "temp1",
    ("temp_and_humidity_ch1", "humidity"): "hmdty1",
    ("battery", "haptic_array_battery"): "haptic_array_battery",
    ("battery", "haptic_array_capacitor"): "haptic_array_capacitor",
    ("battery", "aqi_combo_sensor"): "aqi_battey",
    ("battery", "temp_humidity_sensor_ch1"): "temp1_battery",
    ("battery", "console"): "console_battery"
}

external_default_values = {col: -999 for col in external_column_mapping.values()}
external_default_values["rain_state"] = "0"

# 가평 이수근 봉군 장비용(2025.11.04)
internal_required_data = [
    ("indoor", "temperature"), ("indoor", "humidity"),
    ("pressure", "relative"), ("pressure", "absolute"), ("co2_aqi_combo", "co2"),
    ("pm25_aqi_combo", "real_time_aqi"), ("pm10_aqi_combo", "real_time_aqi"),
    ("t_rh_aqi_combo", "temperature"), ("t_rh_aqi_combo", "humidity"),
    ("temp_and_humidity_ch2", "temperature"), ("temp_and_humidity_ch2", "humidity"), 
    ("temp_and_humidity_ch3", "temperature"), ("temp_and_humidity_ch3", "humidity"), 
    ("temp_and_humidity_ch4", "temperature"), ("temp_and_humidity_ch4", "humidity"), 
    ("temp_and_humidity_ch5", "temperature"), ("temp_and_humidity_ch5", "humidity"), 
    ("battery", "console"), ("battery", "aqi_combo_sensor"),
    ("battery", "temp_humidity_sensor_ch2"), ("battery", "temp_humidity_sensor_ch3"),
    ("battery", "temp_humidity_sensor_ch4"), ("battery", "temp_humidity_sensor_ch5")
]

internal_column_mapping = {
    ("indoor", "temperature"): "indoor_temp",
    ("indoor", "humidity"): "indoor_hmdty",
    ("pressure", "relative"): "pressure_relative",
    ("pressure", "absolute"): "pressure_absolute",
    ("co2_aqi_combo", "co2"): "co2",
    ("pm25_aqi_combo", "real_time_aqi"): "pm25",
    ("pm10_aqi_combo", "real_time_aqi"): "pm10",
    ("t_rh_aqi_combo", "temperature"): "aqi_temp",
    ("t_rh_aqi_combo", "humidity"): "aqi_hmdty",
    ("temp_and_humidity_ch2", "temperature"): "temp2",
    ("temp_and_humidity_ch3", "temperature"): "temp3",
    ("temp_and_humidity_ch4", "temperature"): "temp4",
    ("temp_and_humidity_ch5", "temperature"): "temp5",
    ("temp_and_humidity_ch2", "humidity"): "hmdty2",
    ("temp_and_humidity_ch3", "humidity"): "hmdty3",
    ("temp_and_humidity_ch4", "humidity"): "hmdty4",
    ("temp_and_humidity_ch5", "humidity"): "hmdty5",
    ("battery", "console"): "console_battery",
    ("battery", "aqi_combo_sensor"): "aqi_battey",
    ("battery", "temp_humidity_sensor_ch2"): "temp2_battery",
    ("battery", "temp_humidity_sensor_ch3"): "temp3_battery",
    ("battery", "temp_humidity_sensor_ch4"): "temp4_battery",
    ("battery", "temp_humidity_sensor_ch5"): "temp5_battery"
}

internal_default_values = {col: -999 for col in internal_column_mapping.values()}

# 가평 이수근 강수량 장비용(2025.11.04)
rainfall_required_data = [
    ("outdoor", "temperature"), ("outdoor", "humidity"),
    ("indoor", "temperature"), ("indoor", "humidity"),
    ("rainfall", "rain_rate"), ("rainfall", "daily"),
    ("rainfall", "state"), ("pressure", "relative"),
    ("pressure", "absolute"), ("battery", "console"),
    ("battery", "rainfall_mini_sensor")
]

rainfall_column_mapping = {
    ("outdoor", "temperature"): "outdoor_temp",
    ("outdoor", "humidity"): "outdoor_hmdty",
    ("indoor", "temperature"): "indoor_temp",
    ("indoor", "humidity"): "indoor_hmdty",
    ("rainfall", "rain_rate"): "rain_rate",
    ("rainfall", "daily"): "rain_daily",
    ("rainfall", "state"): "rain_state",
    ("pressure", "relative"): "pressure_relative",
    ("pressure", "absolute"): "pressure_absolute",
    ("battery", "console"): "console_battery",
    ("battery", "rainfall_mini_sensor"): "rainfall_battery"
}

rainfall_default_values = {col: -999 for col in rainfall_column_mapping.values()}
rainfall_default_values["rain_state"] = "0"

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

# 산숲팜 장비용(2025.11.04)
def insert_sansup_ecowitt_data(df, device_id):
    conn, cursor = None, None
    try:
        conn = mysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
            INSERT IGNORE INTO TB_ECOWITT_DATA (
                tm, outdoor_temp, outdoor_hmdty, indoor_temp, indoor_hmdty, 
                solar, uvi, rain_rate, rain_daily, rain_state, 
                wind_speed, wind_direction, pressure_relative, pressure_absolute, co2, 
                pm25, pm10, aqi_temp, aqi_hmdty, temp1,
                temp2, temp3, hmdty1, hmdty2, hmdty3, 
                temp_ch1, haptic_array_battery, haptic_array_capacitor, aqi_battey, temp1_battery, 
                temp2_battery, temp3_battery, temp_ch1_battery, device_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s
            )
        """

        records_to_insert = []
        for _, row in df.iterrows():
            records_to_insert.append(tuple([
                row["tm"], row["outdoor_temp"], row["outdoor_hmdty"], row["indoor_temp"], row["indoor_hmdty"], 
                row["solar"], row["uvi"], row["rain_rate"], row["rain_daily"], row["rain_state"], 
                row["wind_speed"], row["wind_direction"], row["pressure_relative"], row["pressure_absolute"], row["co2"],
                row["pm25"], row["pm10"], row["aqi_temp"], row["aqi_hmdty"], row["temp1"], 
                row["temp2"], row["temp3"], row["hmdty1"], row["hmdty2"], row["hmdty3"], 
                row["temp_ch1"], row["haptic_array_battery"], row["haptic_array_capacitor"], row["aqi_battey"], row["temp1_battery"], 
                row["temp2_battery"], row["temp3_battery"], row["temp_ch1_battery"], device_id
            ]))

        cursor.executemany(query, records_to_insert)
        conn.commit()

        # 2025.09.19
        pd.set_option('display.max_columns', None)
        log.info(f"[CHECK] device_id : {device_id} / final_df : {final_df}")
    except mysql.Error as err:
        if conn: conn.rollback()
        log.error(f"데이터베이스 오류 : {err}")
    except Exception as e:
        if conn: conn.rollback()
        log.error(f"일반 오류 : {e}")
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()


# 가평 이수근 실외 장비용(2025.11.04)
def insert_external_ecowitt_data(df, device_id):
    conn, cursor = None, None
    try:
        conn = mysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
            INSERT IGNORE INTO TB_ECOWITT_DATA (
                tm, outdoor_temp, outdoor_hmdty, solar, uvi,
                rain_rate, rain_daily, rain_state, wind_speed, wind_direction, 
                co2, pm25, pm10, aqi_temp, aqi_hmdty,
                temp1, hmdty1, console_battery, haptic_array_battery, haptic_array_capacitor,
                aqi_battey, temp1_battery, device_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s
            )
        """

        records_to_insert = []
        for _, row in df.iterrows():
            records_to_insert.append(tuple([
                row["tm"], row["outdoor_temp"], row["outdoor_hmdty"], row["solar"], row["uvi"], 
                row["rain_rate"], row["rain_daily"], row["rain_state"], row["wind_speed"], row["wind_direction"],
                row["co2"], row["pm25"], row["pm10"], row["aqi_temp"], row["aqi_hmdty"],
                row["temp1"], row["hmdty1"], row["console_battery"], row["haptic_array_battery"], row["haptic_array_capacitor"], 
                row["aqi_battey"], row["temp1_battery"], device_id
            ]))

        cursor.executemany(query, records_to_insert)
        conn.commit()

        # 2025.09.19
        pd.set_option('display.max_columns', None)
        log.info(f"[CHECK] device_id : {device_id} / final_df : {final_df}")
    except mysql.Error as err:
        if conn: conn.rollback()
        log.error(f"데이터베이스 오류 : {err}")
    except Exception as e:
        if conn: conn.rollback()
        log.error(f"일반 오류 : {e}")
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()


# 가평 이수근 봉군 장비용(2025.11.04)
def insert_internal_ecowitt_data(df, device_id):
    conn, cursor = None, None
    try:
        conn = mysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
            INSERT IGNORE INTO TB_ECOWITT_DATA (
                tm, indoor_temp, indoor_hmdty, pressure_relative, pressure_absolute, 
                co2, pm25, pm10, aqi_temp, aqi_hmdty, 
                temp2, temp3, temp4, temp5, hmdty2, 
                hmdty3, hmdty4, hmdty5, console_battery, aqi_battey, 
                temp2_battery, temp3_battery, temp4_battery, temp5_battery, device_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s
            )
        """

        records_to_insert = []
        for _, row in df.iterrows():
            records_to_insert.append(tuple([
                row["tm"], row["indoor_temp"], row["indoor_hmdty"], row["pressure_relative"], row["pressure_absolute"], 
                row["co2"], row["pm25"], row["pm10"], row["aqi_temp"], row["aqi_hmdty"],
                row["temp2"], row["temp3"], row["temp4"], row["temp5"], row["hmdty2"], 
                row["hmdty3"], row["hmdty4"], row["hmdty5"], row["console_battery"], row["aqi_battey"], 
                row["temp2_battery"], row["temp3_battery"], row["temp4_battery"], row["temp5_battery"], device_id
            ]))

        # 2025.09.19
        pd.set_option('display.max_columns', None)
        log.info(f"[CHECK] device_id : {device_id} / final_df : {final_df}")

        cursor.executemany(query, records_to_insert)
        conn.commit()
    except mysql.Error as err:
        if conn: conn.rollback()
        log.error(f"데이터베이스 오류 : {err}")
    except Exception as e:
        if conn: conn.rollback()
        log.error(f"일반 오류 : {e}")
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()


# 가평 이수근 강수 장비용(2025.11.04)
def insert_rainfall_ecowitt_data(df, device_id):
    conn, cursor = None, None
    try:
        conn = mysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        query = """
            INSERT IGNORE INTO TB_ECOWITT_DATA (
                tm, outdoor_temp, outdoor_hmdty, indoor_temp, indoor_hmdty, 
                rain_rate, rain_daily, rain_state, pressure_relative, pressure_absolute, 
                console_battery, rainfall_battery, device_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s
            )
        """

        records_to_insert = []
        for _, row in df.iterrows():
            records_to_insert.append(tuple([
                row["tm"], row["outdoor_temp"], row["outdoor_hmdty"], row["indoor_temp"], row["indoor_hmdty"], 
                row["rain_rate"], row["rain_daily"], row["rain_state"], row["pressure_relative"], row["pressure_absolute"],
                row["console_battery"], row["rainfall_battery"], device_id
            ]))

        # 2025.09.19
        pd.set_option('display.max_columns', None)
        log.info(f"[CHECK] device_id : {device_id} / final_df : {final_df}")

        cursor.executemany(query, records_to_insert)
        conn.commit()
    except mysql.Error as err:
        if conn: conn.rollback()
        log.error(f"데이터베이스 오류 : {err}")
    except Exception as e:
        if conn: conn.rollback()
        log.error(f"일반 오류 : {e}")
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

# -------------------------
# 메인 실행
# -------------------------
# 2025.11.04 가평 장비 3개 추가
if __name__ == "__main__":
    try:
        mac_id_map = {
            config.get('ecowitt-device1', 'MAC') : config.get('ecowitt-device1', 'DEVICE_ID'),
            config.get('ecowitt-device2', 'MAC') : config.get('ecowitt-device2', 'DEVICE_ID'),
            config.get('ecowitt-device3', 'MAC') : config.get('ecowitt-device3', 'DEVICE_ID'),
            config.get('ecowitt-device4', 'MAC') : config.get('ecowitt-device4', 'DEVICE_ID')
        }
        
        # 각 장비별 매핑 함수 및 데이터 구조
        device_configs = {
            config.get('ecowitt-device1', 'MAC'): {
                "required": sansup_required_data, 
                "mapping": sansup_column_mapping, 
                "default": sansup_default_values,
                "insert_func": insert_sansup_ecowitt_data
            },            
            config.get('ecowitt-device2', 'MAC'): {
                "required": external_required_data, 
                "mapping": external_column_mapping, 
                "default": external_default_values,
                "insert_func": insert_external_ecowitt_data
            },
            config.get('ecowitt-device3', 'MAC'): {
                "required": internal_required_data, 
                "mapping": internal_column_mapping, 
                "default": internal_default_values,
                "insert_func": insert_internal_ecowitt_data
            },
            config.get('ecowitt-device4', 'MAC'): {
                "required": rainfall_required_data, 
                "mapping": rainfall_column_mapping, 
                "default": rainfall_default_values,
                "insert_func": insert_rainfall_ecowitt_data
            }
        }
        
        for mac, device_id in mac_id_map.items():
            in_config = device_configs.get(mac)
            if not in_config:
                log.warning(f"설정되지 않은 MAC 주소 발견: {mac}. 건너뜁니다.")
                continue

            PARAMS = {
                "application_key": APPLICATION_KEY,
                "api_key": API_KEY,
                "mac": mac,
                "temp_unitid": 1,
                "pressure_unitid": 3,
                "wind_speed_unitid": 6
            }

            try:
                response = requests.get(BASE_URL, params=PARAMS)
                
                # API 응답 결과 로깅 (원본 코드에 있던 출력)
                if response.status_code != 200:
                    log.error(f"API 호출 실패 (MAC: {mac}) : {response.status_code}")
                    continue
                
                response_json = response.json()
                
                data = response_json.get("data", {})
                if data is None or len(data) < 1: continue
                if isinstance(data, list) and response_json.get("code") == 40000:
                    log.error(f"MAC: {mac} 에 대해 'Invalid MAC' 오류 응답을 받았습니다. 다음 장비로 넘어갑니다.")
                    continue
                
                df = flatten_sensor_data(data)

                # 데이터 전처리
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                df["datetime_kst"] = pd.to_datetime(df["timestamp"], unit="s", utc=True) + timedelta(hours=9)
                df["value"] = pd.to_numeric(df["value"], errors="coerce")

                # 데이터 필터링 및 컬럼 매핑 적용
                df_filtered = df[df[["category", "sensor"]].apply(tuple, axis=1).isin(in_config["required"])].copy()
                df_filtered["column"] = df_filtered.apply(lambda row: in_config["mapping"].get((row["category"], row["sensor"])), axis=1)

                # 데이터 피벗 및 재정렬
                pivoted_df = df_filtered.pivot_table(
                    index=["timestamp", "datetime_kst"], columns="column", values="value"
                ).reset_index()

                pivoted_df["datetime_kst"] = pd.to_datetime(pivoted_df["datetime_kst"]).dt.strftime("%Y-%m-%d %H:%M:%S")

                final_df = pivoted_df.reindex(columns=["datetime_kst"] + list(in_config["default"].keys()))
                final_df = final_df.rename(columns={"datetime_kst": "tm"}).fillna(value=in_config["default"])

                # DB 저장 함수 호출
                in_config["insert_func"](final_df, device_id)

            except Exception as e:
                log.error(f"스크립트 실행 중 오류 발생 (MAC: {mac}) : {e}")

    except Exception as e:
        log.error(f"스크립트 실행 중 오류 발생 : {e}")
