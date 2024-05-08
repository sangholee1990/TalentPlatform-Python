# /usr/bin/env python


import os
import sys
import copy
import json
import random
import datetime
import numpy as np
import pandas as pd
# import pymysql as mysql       # 추후 DB연동시 필요


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


import threading
lock = threading.Lock()


from flask import Flask, request, render_template, session, redirect, url_for, send_file




# =======================================================================================
# Functions

import gzip, pickle
def save_model(obj, filename):
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(obj, file)



def load_model(filename):
    file = gzip.GzipFile(filename, 'rb')
    obj = pickle.load(file)
    return obj




# =======================================================================================
# Config and Variables

with open('config_server.json', 'rb') as f:
    config = json.loads(f.read().decode())



SERVER_IP = config.get('server_ip')
SERVER_PORT = config.get('server_port')



DATA_DIR = config.get('data_dir')
PD_APTS = pd.read_excel(DATA_DIR + 'tbl_apts.xlsx')
INFO_COLS = [
    # '전용면적', '거래금액',
    'area', 'price',
    '교통', '교육', '주거환경', '편의시설',
    '교통_rel_total', '교육_rel_total', '주거환경_rel_total', '편의시설_rel_total'
]
PD_INFO = PD_APTS[INFO_COLS]



# -------------------------------------------------------------
# scaler transforms
scaler = StandardScaler()
PD_INFO_scaled = scaler.fit_transform(PD_INFO)
COSINE_SCORES = cosine_similarity(PD_INFO_scaled)   # 32GB
# -------------------------------------------------------------



# 로그파일 Writer 생성 (날짜별 생성)
LOG_DIR = 'logs/'
os.makedirs(LOG_DIR, exist_ok=True)



# =============================================================================================
# Load Models
# =============================================================================================
# -------------------
# MODEL_ITEM_BASED
# -------------------
try:
    print('loading MODEL_CF...')
    MODEL_DIVIDE = config.get('model_divide')
    # -----------------------------------------------------------------------
    # 모델 파일 분할 저장시 (저장 및 로딩시 메모리 효율화) : pandas 버전 체크 필요 (1.4에서 저장한 것을 1.3에서 호출 불가. 반대는 가능)
    if MODEL_DIVIDE:
        cnts_per_divide = config.get('cnts_per_divide')
        MODEL_DIVIDE_CNT = len(PD_APTS) // cnts_per_divide
        if MODEL_DIVIDE_CNT * cnts_per_divide < len(PD_APTS):
            MODEL_DIVIDE_CNT += 1
        pd_scores = load_model('weights/rcmdapt_cf_scores.model')
        apt_based_simil_cosine = []
        for cnt in range(MODEL_DIVIDE_CNT):
            model_path_cnt = config.get('model_path').split('.')[0] + '_{}.model'.format('%02d' % cnt)
            print(model_path_cnt)
            apt_cosine = load_model(model_path_cnt)
            if len(apt_based_simil_cosine) == 0:
                apt_based_simil_cosine = apt_cosine
            else:
                apt_based_simil_cosine = pd.concat([apt_based_simil_cosine, apt_cosine], axis=0)
            print('model_shape:', apt_based_simil_cosine.shape)
        #
        MODEL_CF = {
            'cosine_similarity': apt_based_simil_cosine,
            'evaluations': pd_scores
        }
    # -----------------------------------------------------------------------
    # 모델 파일 1개로 저장시
    else:
        MODEL_CF = load_model(config.get('model_path'))
    # -----------------------------------------------------------------------
except Exception as ex:
    print(str(ex))
    sys.exit()



# sys.getsizeof(MODEL_CF)




# =======================================================================================
# 웹서버 기본 모듈 생성

app = Flask(__name__)
app.secret_key = 'any random string'
with app.app_context():
    print(app.name)



# CORS정책 (필요?)
# from flask_cors import CORS, cross_origin
# CORS(app)
# CORS(app, resources={r'*': {'origins': '*'}})
# CORS(app, resources={r'/_api/*': {'origins': 'https://webisfree.com:5000'}})





# =======================================================================================
# 웹서버 functions 정의

@app.route('/recommends_cf', methods=['GET', 'POST'])
def recommends_cf():
    try:
        LOG_FILE = LOG_DIR + 'log_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d'))
        Logger = open(LOG_FILE, 'a', encoding='utf-8')
    except Exception as ex:
        print(str(ex))
    temp = Logger.write('\n{}: {}'.format(datetime.datetime.now(), 'recommends_cf()'))
    Logger.flush()
    #
    #
    results_cf = []
    result_dict = {
        'recommends': {
            'cf': results_cf
        },
        'msg': ''
    }
    if request.method == 'POST':
        # -----------------------------------------
        user_id = request.form.get('user_id')           # Null 허용
        apt_idx = int(request.form.get('apt_idx'))      # Necessary
        rcmd_count = 10
        try:
            rcmd_count = int(request.form.get('rcmd_count'))
        except:
            pass
        #
        # -----------------------------------------
        temp = Logger.write('\n{}: {}'.format(
            datetime.datetime.now(), 'recommends_cf_item: user_id={}, apt_idx={}, rcmd_count={}'.format(
                user_id, apt_idx, rcmd_count)))
        Logger.flush()
        #
        #
        # ====================================================================================================
        # Item-based 추천
        # ====================================================================================================
        try:
            if apt_idx is not None and apt_idx != '':
                # ------------------------------------------------------------
                temp = lock.acquire()
                try:
                    pd_cosine = MODEL_CF.get('cosine_similarity')[apt_idx].sort_values(ascending=False)[1: rcmd_count + 1]
                finally:
                    lock.release()
                # ------------------------------------------------------------
                for index in range(len(pd_cosine)):
                    try:
                        if pd_cosine.values[index] > 0:
                            results_cf.append([
                                int(pd_cosine.index[index]), float(np.round(pd_cosine.values[index], 3))
                                # uint8(255점 만점) -> float으로 변환 (용량 별 차이없음)
                                # int(pd_cosine.index[index]), float(np.round(pd_cosine.values[index] / 255, 3))
                            ])
                    except:
                        pass
                # print('results_cf:', results_cf)
                # ------------------------------------------------------------
        except Exception as ex:
            print(str(ex))
            result_dict['msg'] = '협업필터링(CF) 추천 조회시 에러가 발생하였습니다.'
            temp = Logger.write('\n{}: {}'.format(datetime.datetime.now(), str(ex) + '\n' + result_dict['msg']))
            Logger.flush()
    #
    #
    temp = Logger.write('\n{}: {}'.format(datetime.datetime.now(), str(result_dict)))
    Logger.flush()
    return json.dumps(result_dict, ensure_ascii=False).encode('utf8')





@app.route('/recommends_simil', methods=['GET', 'POST'])
def recommends_simil():
    try:
        LOG_FILE = LOG_DIR + 'log_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d'))
        Logger = open(LOG_FILE, 'a', encoding='utf-8')
    except Exception as ex:
        print(str(ex))
    temp = Logger.write('\n{}: {}'.format(datetime.datetime.now(), 'recommends_simil()'))
    Logger.flush()
    #
    #
    results_simil = []
    result_dict = {
        'recommends': {
            'simil': results_simil
        },
        'msg': ''
    }
    if request.method == 'POST':
        # -----------------------------------------
        user_id = request.form.get('user_id')           # Null 허용
        apt_idx = int(request.form.get('apt_idx'))      # Necessary
        rcmd_count = 10
        try:
            rcmd_count = int(request.form.get('rcmd_count'))
        except:
            pass
        #
        # -----------------------------------------
        temp = Logger.write('\n{}: {}'.format(
            datetime.datetime.now(), 'recommends_simil: user_id={}, apt_idx={}, rcmd_count={}'.format(
                user_id, apt_idx, rcmd_count)))
        Logger.flush()
        #
        #
        # ====================================================================================================
        # Similar Apts 추천
        # ====================================================================================================
        try:
            if apt_idx is not None and apt_idx != '':
                # ------------------------------------------------------------
                # similar apts 조회
                apt_simil = COSINE_SCORES[apt_idx]
                #
                sort_desc = np.argsort(apt_simil)[::-1]
                #
                PD_APTS.iloc[apt_idx].values.tolist()
                for i in range(1, rcmd_count + 1):  # 0: 자기자신
                    rcmd_idx = sort_desc[i]
                    results_simil.append([
                        int(rcmd_idx), float(np.round(apt_simil[rcmd_idx], 3))
                    ])
                # print('results_simil:', results_simil)
                # ------------------------------------------------------------
        except Exception as ex:
            print(str(ex))
            result_dict['msg'] = '유사아파트 추천 조회시 에러가 발생하였습니다.'
            temp = Logger.write('\n{}: {}'.format(datetime.datetime.now(), str(ex) + '\n' + result_dict['msg']))
            Logger.flush()
    #
    #
    temp = Logger.write('\n{}: {}'.format(datetime.datetime.now(), str(result_dict)))
    Logger.flush()
    return json.dumps(result_dict, ensure_ascii=False).encode('utf8')




# ===============================================================================================
@app.route("/reload_models", methods=['GET', 'POST'])
def reload_models():
    try:
        LOG_FILE = LOG_DIR + 'log_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d'))
        Logger = open(LOG_FILE, 'a', encoding='utf-8')
    except Exception as ex:
        print(str(ex))
    temp = Logger.write('\n{}: {}'.format(datetime.datetime.now(), '>>> reload_models()'))
    Logger.flush()
    print('\n>>> reload_models()')
    #
    # -----------------------------------------------------------------------------------
    with open('config.json', 'rb') as f:
        config = json.loads(f.read().decode())
    #
    #
    result_dict = {
        'msg': ''
    }
    # -------------------------------------------------------------------------------
    # MODEL_CF
    MODEL_DIVIDE = config.get('model_divide')
    # -----------------------------------------------------------------------
    # 모델 파일 분할 저장시 (저장 및 로딩시 메모리 효율화) : pandas 버전 체크 필요 (1.3 vs 1.4)
    if MODEL_DIVIDE:
        MODEL_DIVIDE_CNT = config.get('model_divide_cnt')
        pd_scores = load_model('weights/rcmdapt_cf_scores.model')
        apt_based_simil_cosine = []
        for cnt in range(MODEL_DIVIDE_CNT):
            apt_cosine = load_model(config.get('model_path').split('.')[0] + '_{}.model'.format('%02d' % cnt))
            if len(apt_based_simil_cosine) == 0:
                apt_based_simil_cosine = apt_cosine
            else:
                apt_based_simil_cosine = pd.concat([apt_based_simil_cosine, apt_cosine], axis=0)
            print('model_shape:', apt_based_simil_cosine.shape)
        #
        temp = lock.acquire()
        try:
            MODEL_CF = {
                'cosine_similarity': apt_based_simil_cosine,
                'evaluations': pd_scores
            }
            result_dict.update({'msg': 'Success'})
        except Exception as ex:
            print(str(ex))
            result_dict.update({'msg': '모델 reloading시 에러가 발생하였습니다.'})
        finally:
            lock.release()
    # -----------------------------------------------------------------------
    # 모델 파일 1개로 저장시
    else:
        temp = lock.acquire()
        try:
            MODEL_CF = load_model(config.get('model_path'))
            result_dict.update({'msg': 'Success'})
        except Exception as ex:
            print(str(ex))
            result_dict.update({'msg': '모델 reloading시 에러가 발생하였습니다.'})
        finally:
            lock.release()
    # -----------------------------------------------------------------------
    Logger.write('\n{}: {}'.format(datetime.datetime.now(), str(result_dict)))
    Logger.flush()
    return json.dumps(result_dict, ensure_ascii=False).encode('utf8')




# ================================================================================================
# Flask web 실행

if __name__ == '__main__':
    #
    if False:
        # -----------------------------------------------------------
        # Start Model Training thread: 주기적으로 모델 업데이트 할 때만 필요
        # -----------------------------------------------------------
        import rcmdapt_CF as ModelThread
        model_trainer = ModelThread.ModelTrainer()
        model_trainer.daemon = True
        model_trainer.start()
    #
    # Start Webserver
    app.run(host=SERVER_IP, port=SERVER_PORT)  # localhost
    #
    # 종료되면
    print('\nGood Bye~\n\n')


