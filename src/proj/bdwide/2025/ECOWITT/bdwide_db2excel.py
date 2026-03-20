import mysql.connector
import pandas as pd
from datetime import datetime

# 1. 사용자로부터 날짜 입력 받기
# '20251021' 형태의 문자열을 입력받음
try:
    # 사용자에게 YYYYMMDD 형식으로 날짜를 입력받음
    input_date_str = input("조회할 날짜를 YYYYMMDD 형식으로 입력하세요 (예: 20251021): ")
    
    # 입력된 문자열을 datetime 객체로 변환 (형식 체크)
    date_obj = datetime.strptime(input_date_str, '%Y%m%d')
    
    # MySQL 쿼리에 사용할 'YYYY-MM-DD' 형식으로 변환
    TARGET_DATE = date_obj.strftime('%Y-%m-%d')
    
except ValueError:
    print("잘못된 날짜 형식입니다. YYYYMMDD 형식으로 정확히 입력해주세요.")
    exit() # 프로그램 종료

# 1. 데이터베이스 연결 정보 설정
DB_CONFIG = {
    'user': '',
    'password': '',
    'host': 'localhost',
    'database': 'DMS01'
}
TIME_COLUMN_NAME = 'created_at'  # 시간 데이터가 저장된 컬럼 이름
EXCEL_FILE_NAME = f"data_export_{TARGET_DATE}.xlsx"

# 3. 날짜별 조회를 위한 SQL 쿼리 작성
SQL_QUERY = f"""
SELECT *
FROM IOT_DATA_INPUT 
WHERE DATE({TIME_COLUMN_NAME}) = '{TARGET_DATE}';
"""
print(f"\n🔍 실행할 SQL 쿼리:\n{SQL_QUERY}")

try:
    # 4. DB 연결 및 쿼리 실행
    conn = mysql.connector.connect(**DB_CONFIG)
    
    # 조회 결과를 pandas DataFrame으로 바로 가져옴
    df = pd.read_sql(SQL_QUERY, conn)

    # 5. 엑셀 파일로 저장
    if not df.empty:
        df.to_excel(EXCEL_FILE_NAME, index=False)
        print(f"{TARGET_DATE} 데이터 {len(df)}건이 '{EXCEL_FILE_NAME}' 파일로 저장되었습니다.")
    else:
        print(f"{TARGET_DATE} 에 해당하는 데이터가 없습니다. 엑셀 파일은 생성되지 않았습니다.")

except mysql.connector.Error as err:
    print(f"데이터베이스 오류: {err}")
except Exception as e:
    print(f"기타 오류 발생: {e}")
finally:
    # 6. 연결 종료
    if 'conn' in locals() and conn.is_connected():
        conn.close()
        print("🔍 MySQL 연결이 종료되었습니다.")