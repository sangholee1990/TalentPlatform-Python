
import os
import datetime
import pandas as pd
import numpy as np
import re
import pandas as pd



gu_list = [
    '강남구', '강동구', '강북구', '중구',                 # 456, 396, 114, 102
    '강서구', '과천시', '관악구', '광진구', '종로구',       # 518, 19, 226, 186, 103
    '구로구', '중랑구', '금천구', '노원구',                # 446, 232, 135, 305
    '도봉구', '동대문구', '동작구', '마포구',              # 205, 248, 210, 286,
    '서대문구', '서초구', '성동구', '성북구',              # 233, 488, 149, 171,
    '송파구', '양천구', '영등포구',                       # 327, 513, 255,
    '용산구', '은평구'                                  # 183, 424,
]
gu_list.sort()



# =========================================================================================
# Functions

import gzip, pickle


def save_model(obj, filename):
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(obj, file)



def load_model(filename):
    file = gzip.GzipFile(filename, 'rb')
    obj = pickle.load(file)
    return obj






# =============================================================================================
# 1. 사용자 정보

'''
gender = 1          # 1: 남자(52.1%), 2: 여자(47.9%)
age = 30            # 서울시 인구현황? 아파트소유자수? 비중
debt_ratio = 30     # 주택구입시 대출비율 (성별, 연령대별, 지역별)
'''

if True:
    # -------------------------------
    CNT_USERS = 5000
    # -------------------------------
    user_list = []
    #
    for i in range(CNT_USERS):
        user_id = 'user_{}'.format('%05d' % i)
        user_name = user_id
        #
        gender = 1
        age = 20 + np.random.randint(10)  # 20~29
        gu_name = gu_list[np.random.randint(len(gu_list))]
        #
        price_from = 0
        price_to = 0.59
        area_from = 0
        area_to = 39.9
        debt_ratio = 0.05
        prefer = '교통'
        #
        rand_gender = np.random.rand()
        rand_age = np.random.rand()
        rand_price = np.random.rand()
        rand_area = np.random.rand()
        rand_debt = np.random.rand()
        #
        # --------------------------------------------------
        # [남자]
        # - 연령
        if rand_age > 0.965:
            age = 80 + np.random.randint(20)  # 80~99
            prefer = '주거환경'
        elif rand_age > 0.860:
            age = 70 + np.random.randint(10)  # 70~79
            prefer = '주거환경'
        elif rand_age > 0.650:
            age = 60 + np.random.randint(10)  # 60~69
            prefer = '편의시설'
        elif rand_age > 0.397:
            age = 50 + np.random.randint(10)  # 50~59
            prefer = '편의시설'
        elif rand_age > 0.149:
            age = 40 + np.random.randint(10)  # 40~49
            prefer = '교육'
        elif rand_age > 0.017:
            age = 30 + np.random.randint(10)  # 30~39
        #
        # --------------------------------------------------
        # - 자산가액 (남녀구분 없음)
        if rand_price > 0.993 or rand_age > 0.860:
            price_from = 12.00
            price_to = 100.00
        elif rand_price > 0.959 or rand_age > 0.650:
            price_from = 6.00
            price_to = 11.99
        elif rand_price > 0.800 or rand_age > 0.397:
            price_from = 3.00
            price_to = 5.99
        elif rand_price > 0.526 or rand_age > 0.149:
            price_from = 1.50
            price_to = 2.99
        elif rand_price > 0.214 or rand_age > 0.017:
            price_from = 0.60
            price_to = 1.49
        #
        # --------------------------------------------------
        # - 면적 (남녀구분 없음)
        if rand_area > 0.969:
            area_from = 165.0
            area_to = 1000.0
        elif rand_area > 0.892:
            area_from = 100.0
            area_to = 164.9
        elif rand_area > 0.512:
            area_from = 60.0
            area_to = 99.9
        elif rand_area > 0.129:
            area_from = 40.0
            area_to = 59.9
        #
        # --------------------------------------------------
        # - 대출비율 (남녀구분 없음)
        if rand_debt > 0.773:
            debt_ratio = 0
        elif rand_debt > 0.761:
            debt_ratio = 0.75
        elif rand_debt > 0.684:
            debt_ratio = 0.45
        elif rand_debt > 0.513:
            debt_ratio = 0.35
        elif rand_debt > 0.209:
            debt_ratio = 0.25
        elif rand_debt > 0.043:
            debt_ratio = 0.15
        #
        # --------------------------------------------------
        # [여자]
        # - 연령
        if rand_gender > 0.521:
            gender = 2
            if rand_age > 0.959:
                age = 80 + np.random.randint(20)  # 80~99
            elif rand_age > 0.858:
                age = 70 + np.random.randint(10)  # 70~79
            elif rand_age > 0.648:
                age = 60 + np.random.randint(10)  # 60~69
            elif rand_age > 0.391:
                age = 50 + np.random.randint(10)  # 50~59
            elif rand_age > 0.149:
                age = 40 + np.random.randint(10)  # 40~49
            elif rand_age > 0.021:
                age = 30 + np.random.randint(10)  # 30~39
        # --------------------------------------------------
        #
        user_list.append([
            user_id, user_name, gender, age, gu_name,
            price_from, price_to, area_from, area_to, debt_ratio, prefer
        ])
    #
    pd_users = pd.DataFrame(user_list)
    pd_users.columns = [
        'user_id', 'name', 'gender', 'age', 'gu_name',
        'price_from', 'price_to', 'area_from', 'area_to', 'debt_ratio', 'prefer'
    ]
    pd_users.to_excel('dataset/tbl_users.xlsx', index=False)




# --------------------------------------------------------------------------------------------
# 2. 아파트 정보

if True:
    pd_surr = pd.read_excel('dataset/surroundings_scores_total.xlsx')
    #
    apts_merge = []
    for gu_name in gu_list:
        print()
        print(gu_name)
        pd_apts = pd.read_excel('dataset/{}실거래가.xlsx'.format(gu_name))
        #
        pd_apts2 = pd_apts[pd_apts.계약년월 >= 202101]
        # pd_apts2 = pd_apts2[['아파트', 'latitude', 'longitude', '전용면적(㎡)', '거래 금액(억원)', '계약년월']].drop_duplicates().reset_index(drop=True)
        pd_apts2 = pd_apts2[['아파트', 'latitude', 'longitude', '전용면적(㎡)', '거래 금액']].drop_duplicates().reset_index(drop=True)  # , '계약년월'
        pd_apts2.columns = ['아파트', 'latitude', 'longitude', '전용면적', '거래금액']     # , '계약년월']
        print(pd_apts2.shape)
        #
        #
        for index, row in pd_apts2.iterrows():
            apt = row['아파트']
            latitude = row['latitude']
            longitude = row['longitude']
            area = row['전용면적']
            price = int(row['거래금액']) / 10**8     # 억원
            # yyyymm = row['계약년월']
            #
            cnt_traf = -1
            cnt_edu = -1
            cnt_env = -1
            cnt_conv = -1
            score_traf = -1
            score_edu = -1
            score_env = -1
            score_conv = -1
            #
            pd_surr_apt = pd_surr[pd_surr.apt_addr==apt][pd_surr.apt_x==longitude][pd_surr.apt_y==latitude] # [pd_surr.apt_area==area]
            if len(pd_surr_apt) > 0:
                cnt_traf = pd_surr_apt['교통'].values[0]
                cnt_edu = pd_surr_apt['교육'].values[0]
                cnt_env = pd_surr_apt['주거환경'].values[0]
                cnt_conv = pd_surr_apt['편의시설'].values[0]
                score_traf = pd_surr_apt['교통_rel'].values[0]
                score_edu = pd_surr_apt['교육_rel'].values[0]
                score_env = pd_surr_apt['주거환경_rel'].values[0]
                score_conv = pd_surr_apt['편의시설_rel'].values[0]
                #
                score_traf_total = pd_surr_apt['교통_rel_total'].values[0]
                score_edu_total = pd_surr_apt['교육_rel_total'].values[0]
                score_env_total = pd_surr_apt['주거환경_rel_total'].values[0]
                score_conv_total = pd_surr_apt['편의시설_rel_total'].values[0]
            #
            apts_merge.append([
                gu_name, apt, latitude, longitude, area, price,  # yyyymm,
                cnt_traf, cnt_edu, cnt_env, cnt_conv,
                score_traf, score_edu, score_env, score_conv,
                score_traf_total, score_edu_total, score_env_total, score_conv_total
            ])
    #
    pd_merge = pd.DataFrame(apts_merge)
    print()
    print('통합:')
    print(pd_merge.shape)
    pd_merge = pd_merge.drop_duplicates()
    print(pd_merge.shape)
    pd_merge.columns = [
        'gu_name', 'apt', 'latitude', 'longitude', 'area', 'price', # '계약년월',
        '교통', '교육', '주거환경', '편의시설',
        '교통_rel', '교육_rel', '주거환경_rel', '편의시설_rel',
        '교통_rel_total', '교육_rel_total', '주거환경_rel_total', '편의시설_rel_total'
    ]
    pd_merge.to_excel('dataset/tbl_apts.xlsx', index=True, index_label='apt_idx')





# --------------------------------------------------------------------------------------------
# 3. 사용자별 아파트 평가정보 (조회, 라이크, 평점)

if True:
    pd_users = pd.read_excel('dataset/tbl_users.xlsx')
    pd_apts = pd.read_excel('dataset/tbl_apts.xlsx')
    # -------------------------------
    # EVALS_PER_USER = 1000
    EVALS_PER_USER = 100
    # -------------------------------
    #
    user_apt_evals = []
    # 사용자별로 100개 아파트 조회
    # - 조회한 아파트에 대해 30%만 라이크
    # - 조회한 아파트 평가는 50%만 (1~10점, 좋아요 클릭시 6~10점, 좋아요 미클릭시 1~5점 랜덤)
    for idx, row in pd_users.iterrows():
        user_id = row['user_id']
        print(user_id)
        price_from = row['price_from']
        price_to = row['price_to']
        area_from = row['area_from']
        area_to = row['area_to']
        prefer = row['prefer']
        #
        #
        for i in range(EVALS_PER_USER):
            apt_idx = apt_list.pop(np.random.randint(len(apt_list)))
            apt_info = pd_apts.loc[apt_idx]
            apt_area = apt_info['area']
            apt_price = apt_info['price']
            #
            #
            # 라이크 여부
            like_yn = 0
            # if np.random.rand() >= 0.7:
            if (apt_price >= price_from and apt_price < price_to) and (apt_area >= area_from and apt_area < area_to):
                like_yn = 1
            #
            # 평가점수
            if like_yn:
                eval_score = 5 + np.random.randint(3)
                #
                if (prefer == '교통' and apt_info['교통_rel'] > 0.9) \
                        or (prefer == '교육' and apt_info['교육_rel'] > 0.9) \
                        or (prefer == '주거환경' and apt_info['주거환경_rel'] > 0.9) \
                        or (prefer == '편의시설' and apt_info['편의시설_rel'] > 0.9):
                    eval_score += 3
            else:
                eval_score = np.random.randint(4)
            #
            user_apt_evals.append([user_id, apt_idx, 1, like_yn, eval_score])
    #
    pd_evals = pd.DataFrame(user_apt_evals)
    pd_evals.columns = ['user_id', 'apt_idx', 'view', 'like', 'evals']  # eval: method 이름이라 사용 불가
    # pd_evals.to_excel('dataset/tbl_user_apt_evals.xlsx', index=False)
    save_model(pd_evals, 'dataset/tbl_user_apt_evals.dat')




