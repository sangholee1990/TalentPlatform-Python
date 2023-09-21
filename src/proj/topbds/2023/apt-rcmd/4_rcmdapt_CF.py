


import os
import json
import shutil
import threading
import time
import datetime
import pymysql as mysql
import pandas as pd
import numpy as np
import requests
import gc
import multiprocessing



from sklearn.metrics.pairwise import cosine_similarity



# =========================================================================================
# config

with open('config_server.json', 'rb') as f:
    config = json.loads(f.read().decode())

print(config)


DATA_DIR = config.get('data_dir')
os.makedirs(DATA_DIR, exist_ok=True)


wgts_view = config.get('wgts_view')
wgts_eval = config.get('wgts_eval')


# -------------------------------------------
MULTIPROCESSING = True
# -------------------------------------------


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



# =========================================================================================

def user_scoring(start_idx, end_idx, user_list, apt_list, pd_evals):
    #
    user_apt_scores = []
    for user_id in user_list:
        start = time.time()
        apt_scores = np.zeros(len(apt_list), dtype='uint8')
        #
        pd_user_id = pd_evals[pd_evals.user_id == user_id]
        user_apt_list = pd_user_id['apt_idx'].tolist()
        # print(1, time.time() - start)
        start = time.time()
        #
        for apt_idx in user_apt_list:
            score_total = 0
            try:
                # start = time.time()
                # pd_score = pd_evals[pd_evals.user_id == user_id][pd_evals.apt_idx == apt_idx]
                pd_score = pd_user_id[pd_user_id.apt_idx == apt_idx]
                # print(2, time.time() - start)
                score_view = pd_score['view'].values[0] * wgts_view
                score_eval = pd_score['evals'].values[0] * wgts_eval    # config.get('wgts_eval')
                score_total = score_view + score_eval
                # print('\t', user_id, apt_idx, score_total)
            except Exception as ex:
                print(str(ex))
            apt_scores[apt_idx] = score_total
        #
        apt_scores = list(apt_scores)
        print(user_id, np.unique(apt_scores), time.time() - start)
        user_apt_scores.append([user_id] + apt_scores)
    #
    pd_scores = pd.DataFrame(user_apt_scores)
    pd_scores.columns = ['user_id'] + apt_list
    save_model(pd_scores, 'dataset/user_apt_scores_{}_{}.dat'.format(
        '%06d' % start_idx, '%06d' % end_idx))





# ============================================================================================
class ModelTrainer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.IS_RUN = True
        self.sleep_time = 0.1
    #
    #
    def run(self):
        while self.IS_RUN:
            print()
            print('ModelTrainer IS RUNNING')
            # try:
            if True:
                start = time.time()
                if False:
                    # ---------------------------------------------------------------------
                    # 주기적 업데이트 실행시 필요
                    try:
                        self.sleep_time = int(config.get('sleep_days') * 24 * 60 * 60)
                        print('read (db/file) every {}seconds'.format(self.sleep_time))
                    except Exception as ex:
                        print(str(ex))
                    # ---------------------------------------------------------------------
                #
                #
                # =================================================================================
                # 파일정보
                # =================================================================================
                # 사용자
                pd_users = pd.read_excel(DATA_DIR + 'tbl_users.xlsx')
                print('pd_users:', pd_users.shape)
                user_list = pd_users['user_id'].drop_duplicates().tolist()
                #
                #
                # 아파트
                pd_apts = pd.read_excel(DATA_DIR + 'tbl_apts.xlsx')
                print('pd_apts:', pd_apts.shape)
                apt_list = pd_apts.index.tolist()
                #
                #
                # 평가정보
                pd_evals = load_model(DATA_DIR + 'tbl_user_apt_evals.dat')
                print('pd_evals:', pd_evals.shape)
                #
                # ---------------------------------------------------------------------------------
                #
                CNT_USER = config.get('cnts_per_scoring')
                CNT_SCORER = len(user_list) // CNT_USER
                if CNT_USER * CNT_SCORER < len(user_list):
                    CNT_SCORER += 1
                print('CNT_USER:', CNT_USER, ', CNT_SCORER:', CNT_SCORER)
                #
                # ---------------------------------------------------------------------------------
                if True:
                    if MULTIPROCESSING:
                        procs = []
                        for cnt in range(CNT_SCORER):
                            start_idx = CNT_USER * cnt
                            end_idx = CNT_USER * (cnt + 1)
                            #
                            p = multiprocessing.Process(
                                target=user_scoring,
                                args=(start_idx, end_idx, user_list[start_idx: end_idx], apt_list, pd_evals)
                            )
                            p.start()
                            procs.append(p)
                            time.sleep(1)
                        #
                        for p in procs:
                            p.join()  # 프로세스가 모두 종료될 때까지 대기
                    else:
                        for cnt in range(CNT_SCORER):
                            start_idx = CNT_USER * cnt
                            end_idx = CNT_USER * (cnt + 1)
                            user_scoring(start_idx, end_idx, user_list[start_idx: end_idx], apt_list, pd_evals)
                #
                print('소요시간:', time.time() - start)
                #
                # ---------------------------------------------------------------------------------
                if True:
                    pd_scores = []
                    for i in range(CNT_SCORER):
                        try:
                            start_idx = CNT_USER * i
                            end_idx = CNT_USER * (i + 1)
                            user_score_path = DATA_DIR + 'user_apt_scores_{}_{}.dat'.format(
                                '%06d' % start_idx, '%06d' % end_idx)
                            pd_part = load_model(user_score_path)
                            #
                            if len(pd_scores) == 0:
                                pd_scores = pd_part
                            else:
                                pd_scores = pd.concat([pd_scores, pd_part], axis=0)
                            # -----------------------------
                            # os.remove(user_score_path)
                            # -----------------------------
                        except Exception as ex:
                            print(str(ex))
                    print(pd_scores.shape)
                    #
                    # =====================
                    # Item-based 협업추천
                    # =====================
                    if len(pd_scores) != 0:
                        # Cosine Similarity
                        apt_user_rating = pd_scores.values[:, 1:]
                        apt_user_rating = apt_user_rating.transpose()
                        apt_user_rating = apt_user_rating.astype('uint8')
                        apt_based_simil_cosine = cosine_similarity(apt_user_rating)
                        # ------------------------------------------------------------
                        # float -> uint8 로 변경
                        # apt_based_simil_cosine = (apt_based_simil_cosine * 255).astype('uint8') # 용량줄이기(별 차이없음)
                        # ------------------------------------------------------------
                        apt_based_simil_cosine = pd.DataFrame(
                            data=apt_based_simil_cosine,
                            index=pd_apts.index,
                            columns=pd_apts.index
                        )
                        print('apt_based_simil_cosine:', apt_based_simil_cosine.shape)
                        # ----------------------
                        # Save (backup)
                        # ----------------------
                        if True:
                            MODEL_DIVIDE = config.get('model_divide')
                            cnts_per_divide = 5000  # default
                            try:
                                cnts_per_divide = int(config.get('cnts_per_divide'))
                            except:
                                cnts_per_divide = 5000
                            #
                            model_path = config.get('model_path')
                            #
                            if MODEL_DIVIDE:
                                try:
                                    save_model(pd_scores, 'weights/rcmdapt_cf_scores.model')
                                    #
                                    MODEL_DIVIDE_CNT = len(apt_based_simil_cosine) // cnts_per_divide
                                    if cnts_per_divide * MODEL_DIVIDE_CNT < len(apt_based_simil_cosine):
                                        MODEL_DIVIDE_CNT += 1
                                    for cnt in range(MODEL_DIVIDE_CNT):
                                        start = time.time()
                                        start_idx = cnts_per_divide * cnt
                                        end_idx = cnts_per_divide * (cnt + 1)
                                        cosine_cnt = apt_based_simil_cosine[start_idx: end_idx]
                                        print(cnt, start_idx, end_idx, cosine_cnt.shape)
                                        #
                                        # 이전 모델 백업 Backup
                                        model_path_cnt = config.get('model_path').split('.')[0] + '_{}.model'.format('%02d' % cnt)
                                        try:
                                            # 기존 파일은 -> 직전파일로
                                            src = model_path_cnt
                                            dst = model_path_cnt + '_before'
                                            shutil.copy(src, dst)
                                            print('>>> copied:', dst)
                                        except Exception as ex:
                                            print(str(ex))
                                        #
                                        try:
                                            # save_model(cosine_cnt, model_path_cnt)
                                            print('>>> saved:', model_path_cnt, cnt, time.time() - start)
                                        except Exception as ex:
                                            print(str(ex))
                                except Exception as ex:
                                    print(str(ex))
                            #
                            # Model 분할 안함
                            else:
                                MODEL_CF = {
                                    'cosine_similarity': apt_based_simil_cosine,
                                    'evaluations': pd_scores
                                }
                                # 이전 모델 백업 Backup
                                try:
                                    # 기존 파일은 -> 직전파일로
                                    src = model_path
                                    dst = model_path + '_before'
                                    shutil.copy(src, dst)
                                    print('>>> copied:', dst)
                                except Exception as ex:
                                    print(str(ex))
                                #
                                try:
                                    save_model(MODEL_CF, model_path)
                                    print('>>> saved:', model_path)
                                    gc.collect()
                                except Exception as ex:
                                    print(str(ex))
                    #
                    gc.collect()
                    end = time.time()
                    print('소요시간:', end - start)
                #
                # ============================================================================================
                # Reload Models
                if False:
                    try:
                        url = 'http://{}:{}/reload_models'.format( config['server_ip'], config['server_port'] )
                        res = requests.post(url)
                        print('\nres_json:', res.json())
                    except Exception as ex:
                        print(str(ex))
                #
                # ============================================================================================
                # Sleeping
                time.sleep(self.sleep_time)
            # except Exception as ex:
            #    print(str(ex))
            # -----------------------------
            self.IS_RUN = False
            # -----------------------------



# ========================================================================
# 자체 실행


def main():
    # your training code
    db_th = ModelTrainer()
    # db_th.daemon = True
    db_th.start()
    # db_th.run()       # no_threading



if __name__=='__main__':
    main()


# nohup python 4_rcmdapt_CF.py   output.log 2&1 &

