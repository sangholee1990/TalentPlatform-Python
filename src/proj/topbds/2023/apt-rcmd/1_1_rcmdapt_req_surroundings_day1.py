

import os
import numpy as np
import pandas as pd
import requests
import xmltodict



# =====================================================================================
# Variables

KAKAO_API = 'https://dapi.kakao.com/v2/local/search/keyword.json?&query={}&category_group_code={}&y={}&x={}&radius={}'
KAKAO_API_KEY = 'be53c18774a32321d32ae334065b3dee'   # BDS TOP
# 1일 약 5천개 가능 (2일로 나누어 작업 필요)



BUS_API_BASE = 'http://ws.bus.go.kr/api/rest/stationinfo/getStationByPos?serviceKey={}&tmX={}&tmY={}&radius={}'
BUS_API_KEY = 'LO%2FPM7UaWcr9d2ypUzQzjYK3KyEoRZtKIKuvrDfhg51mXlk2eFriN%2FWCZV249GrQorog5Fv12iOUeWx9nhwDWA%3D%3D' # BDS TOP
# 1일 약 1만개 가능


CATEGORIES = {
    '교통': [('지하철역', 'SW8')],    # 버스정류장 관련정보(X) -> 별도 API로 조회
    '교육': [('학교', 'SC4'), ('학원', 'AC5'), ('어린이집', 'PS3'), ('유치원', 'PS3')],
    '주거환경': [('주차장', 'PK6'), ('주유소', 'OL7'), ('충전소', 'OL7'), ('문화시설', 'CT1'), ('공공기관', 'PO3'), ('관광명소', 'AT4')],    # 공원(X)
    '편의시설': [('병원', 'HP8'), ('약국', 'PM9'), ('은행', 'BK9'), ('대형마트', 'MT1'), ('편의점', 'CS2'), ('음식점', 'FD6'), ('까페', 'CE7')]
}



# =====================================================================================
# 아파트 실거래가 데이터

data_dir = 'dataset/'

# -----------------------------------
# bus_api: 1일 1만개 가능
# kakao_api: 1일 약 5천개
gu_list = [
    # 1일
    '강남구', '강동구', '강북구', '중구',                 # 456, 396, 114, 102
    '강서구', '과천시', '관악구', '광진구', '종로구',       # 518, 19, 226, 186, 103
    '구로구', '중랑구', '금천구', '노원구',                # 446, 232, 135, 305
    '도봉구', '동대문구', '동작구', '마포구',              # 205, 248, 210, 286,
    #
    # 2일
    # '서대문구', '서초구', '성동구', '성북구',              # 233, 488, 149, 171,
    # '송파구', '양천구', '영등포구',                       # 327, 513, 255,
    # '용산구', '은평구'                                  # 183, 424,
]
# -----------------------------------



for gu_name in gu_list:
    print()
    apt_file = '{}실거래가.xlsx'.format(gu_name)
    #
    PD_APTS = pd.read_excel(data_dir + apt_file)
    PD_APTS = PD_APTS[['아파트', 'latitude', 'longitude']].drop_duplicates()
    print(gu_name, PD_APTS.shape)
    BROKEN = False
    #
    #
    # --------------------------------------------
    # Read data & query surroundings info (위치 정보만으로 조회:  '아파트', 'latitude', 'longitude')
    results = []
    xy_list = []
    for index, row in PD_APTS.iterrows():
        # if gu_name == '강북구' and index < 707:
        #    continue
        # ---------------------------------------------------
        # gu_name
        apt = row['아파트']
        y = row['latitude']
        x = row['longitude']
        # area = row['전용면적']    # x, y만으로 주변시설 조회
        radius = 1000
        x_y = '{}_{}'.format(x, y)
        # ---------------------------------------------------
        # 중복조회 방지
        if x_y in xy_list:
            continue
        elif str(x) == 'nan' or str(y) == 'nan':
            continue
        else:
            xy_list.append(x_y)
        print('\tlong:', x, ' ,lati:', y)
        #
        # ---------------------------------
        # kakao info
        # ---------------------------------
        for category, query_codes in CATEGORIES.items():
            print('\t\tcategory:', category)
            #
            for query, group_code in query_codes:
                headers = {'Authorization': 'KakaoAK {}'.format(KAKAO_API_KEY)}
                req_url = KAKAO_API.format(query, group_code, y, x, radius)
                #
                req_res = requests.get(req_url, headers=headers)
                res_json = req_res.json()
                if 'API limit has been exceeded' in str(res_json):
                    BROKEN = True
                    break
                # print('\t\t', query, res_json.get('meta').get('total_count'))
                #
                for doc in res_json.get('documents'):
                    c_g_code = doc.get('category_group_code')
                    c_g_name = doc.get('category_group_name')
                    p_name = doc.get('place_name')
                    p_x = float(doc.get('x'))
                    p_y = float(doc.get('y'))
                    p_dist = -1
                    try:
                        p_dist = int(doc.get('distance'))
                    except:
                        pass
                    p_addr = doc.get('address_name')
                    p_road_addr = doc.get('road_address_name')
                    p_phone = doc.get('phone')
                    p_url = doc.get('place_url')
                    results.append([
                        gu_name, apt, x, y, radius, category, query, c_g_code, c_g_name,
                        p_name, p_dist, p_x, p_y, p_addr, p_road_addr, p_phone, p_url
                    ])
        #
        # -------------------------------------------------------------------
        # bus-stop info (500미터 거리내 버스정류장)
        # -------------------------------------------------------------------
        try:
            print('\t\tcategory: 교통_버스정류장')
            BUS_API = BUS_API_BASE.format(BUS_API_KEY, x, y, 1000)
            bus_res = requests.get(BUS_API)
            # output = bus_res.json()
            # print(output)
            #
            res_json = xmltodict.parse(bus_res.text)
            if 'LIMITED NUMBER OF SERVICE REQUESTS EXCEEDS' in str(res_json):
                BROKEN = True
                break
            items = res_json.get('ServiceResult').get('msgBody').get('itemList')
            #
            for item in items:
                b_y = item.get('gpsY')
                b_x = item.get('gpsX')
                nodenm = item.get('stationNm')
                nodeno = item.get('stationId')
                b_dist = -1
                try:
                    b_dist = int(item.get('dist'))
                except:
                    pass
                #
                b_addr, b_road_addr, b_phone, b_url = '', '', '', ''
                if b_dist <= 1000:
                    # print('\t\t\t', y, x, nodeno, nodenm, b_dist)
                    results.append([
                        gu_name, apt, x, y, 1000, '교통', '버스정류장', '', '버스정류장',
                        '{}_{}'.format(nodeno, nodenm), b_dist, b_x, b_y, b_addr, b_road_addr, b_phone, b_url
                    ])
        except Exception as ex:
            print(str(ex))
    #
    # gu_name별로 저장
    if not BROKEN:
        pd_results = pd.DataFrame(results)
        pd_results.columns = [
            'gu_name', 'apt', 'apt_x', 'apt_y', 'radius', 'category', 'query', 'c_g_code', 'c_g_name',
            'place_name', 'distance', 'p_x', 'p_y', 'p_addr', 'p_road_addr', 'p_phone', 'p_url'
        ]
        print(pd_results.shape)
        pd_results.to_excel('dataset/surroundings_{}.xlsx'.format(gu_name), index=False)


