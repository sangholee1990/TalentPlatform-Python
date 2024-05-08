# -*- coding: utf-8 -*-


import os
import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from glob import glob
import time
import copy
import shutil
import cv2
from PIL import Image
import json
import requests



from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui



# =======================================================================================
# Functions

import gzip, pickle


# ==================================================================================================
# Functions

def save_model(obj, filename):
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(obj, file)



def load_model(filename):
    file = gzip.GzipFile(filename, 'rb')
    obj = pickle.load(file)
    return obj



# ==================================================================================================
# Variables

with open('./config_server.json', 'rb') as f:
    config = json.loads(f.read().decode())



SERVER_IP = config['server_ip']
PORT = config['server_port']
REST_API_URL_CF = 'http://{}:{}/recommends_cf'.format(SERVER_IP, PORT)
REST_API_URL_SIMIL = 'http://{}:{}/recommends_simil'.format(SERVER_IP, PORT)


WIDTH = 1920
try:
    WIDTH = config.get('client_width')
except:
    WIDTH = 1920


HEIGHT = 1400
try:
    HEIGHT = config.get('client_height')
except:
    HEIGHT = 1400


WID_COMP = 200
try:
    WID_COMP = config.get('client_comp_width')
except:
    WID_COMP = 200


# -----------------------------------------------------
# pd_evals = pd.read_excel('dataset/tbl_user_apt_evals.xlsx')
PD_USERS = pd.read_excel('dataset/tbl_users.xlsx')
# -------------------------------------------------
# 희망 가격 6억원대 이상 고객군에 한함
PD_USERS = PD_USERS[PD_USERS.price_from >= config.get('target_cus_price')][PD_USERS.area_from >= 60]
USER_GU_LIST = PD_USERS['gu_name'].drop_duplicates().tolist()
USER_GU_LIST.sort()
# USER_LIST = PD_USERS['user_id'].drop_duplicates().tolist()
# USER_LIST.sort()
# -------------------------------------------------

PD_APTS = pd.read_excel('dataset/tbl_apts.xlsx')
APT_LIST = PD_APTS['apt_idx'].astype('str').tolist()


# 아파트 환경 정보
PD_APT_SURR = pd.read_excel('dataset/surroundings_scores_total.xlsx')
PD_EVALS = load_model('dataset/tbl_user_apt_evals.dat')




# ==================================================================================================
# Class

class AlignDelegate(QtWidgets.QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super(AlignDelegate, self).initStyleOption(option, index)
        option.displayAlignment = QtCore.Qt.AlignCenter


class MyComboBox(QComboBox):
    def __init__(self, parent=None, scrollWidget=None, *args, **kwargs):
        super(MyComboBox, self).__init__(parent, *args, **kwargs)
        self.scrollWidget=scrollWidget
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
    #
    def wheelEvent(self, *args, **kwargs):
        return



# =======================================================================================================
# Main Window
class AptClient(QMainWindow):
    def __init__(self):
        super(AptClient, self).__init__()
        try:
            # 윈도우 특성 설정
            self.setWindowTitle('AptClient')  # 윈도우 타이클 지정
            self.setGeometry(0, 0, WIDTH, HEIGHT)  # 윈도우 위치/크기 설정
            # ===================================================================
            # Object
            self.frame1 = QFrame(self)
            self.frame1.setGeometry(QtCore.QRect(0, 0, WIDTH, HEIGHT))
            self.frame1.setFrameShape(QFrame.StyledPanel)
            self.frame1.setFrameShadow(QFrame.Raised)
            #
            # -----------------------------------------------------------------
            # User
            self.label_user = QLabel('사용자선택', self.frame1)
            self.label_user.setGeometry(QRect(0, 0, WID_COMP, 30))
            self.label_user.setAlignment(Qt.AlignCenter)
            #
            self.combo_user_gu = QComboBox(self.frame1)
            self.combo_user_gu.setGeometry(QtCore.QRect(WID_COMP, 0, WID_COMP - 10, 30))
            self.combo_user_gu.addItems(USER_GU_LIST)
            self.combo_user_gu.currentTextChanged.connect(self.user_gu_changed)
            #
            self.combo_user = QComboBox(self.frame1)
            self.combo_user.setGeometry(QtCore.QRect(WID_COMP * 2, 0, WID_COMP, 30))
            # self.combo_user.addItems(USER_LIST)
            self.combo_user.currentTextChanged.connect(self.user_changed)
            #
            self.label_user_info = QLabel('', self.frame1)
            self.label_user_info.setGeometry(QRect(10, 40, WID_COMP * 3, 200))
            #
            #
            # -----------------------------------------------------------------
            # Apt
            self.label_apts = QLabel('조회한 아파트 선택', self.frame1)
            self.label_apts.setGeometry(QRect(50 + WID_COMP * 3, 0, WID_COMP, 30))
            self.label_apts.setAlignment(Qt.AlignCenter)
            #
            self.combo_apts_gu = QComboBox(self.frame1)
            self.combo_apts_gu.setGeometry(QtCore.QRect(50 + WID_COMP * 4, 0, WID_COMP - 10, 30))
            # self.combo_apts_gu.addItems(APT_GU_LIST)
            self.combo_apts_gu.currentTextChanged.connect(self.apt_gu_changed)
            #
            self.combo_apts = QComboBox(self.frame1)
            self.combo_apts.setGeometry(QtCore.QRect(50 + WID_COMP * 5, 0, WID_COMP, 30))
            # self.combo_apts.addItems(APT_LIST)
            self.combo_apts.currentTextChanged.connect(self.apt_changed)
            #
            self.label_apt_info = QLabel('', self.frame1)
            self.label_apt_info.setGeometry(QRect(50 + WID_COMP * 3, 40, WID_COMP * 3, 200))
            #
            #
            # -----------------------------------------------------------------
            # Recommends
            self.btn_recommends_cf = QPushButton('CF 추천', self.frame1)
            self.btn_recommends_cf.setGeometry(QtCore.QRect(WID_COMP * 7 - 50, 0, WID_COMP - 10, 30))
            self.btn_recommends_cf.clicked.connect(self.recommends_cf)
            #
            self.btn_recommends_simil = QPushButton('유사아파트 추천', self.frame1)
            self.btn_recommends_simil.setGeometry(QtCore.QRect(WID_COMP * 8 - 50, 0, WID_COMP - 10, 30))
            self.btn_recommends_simil.clicked.connect(self.recommends_simil)
            #
            # -----------------------------------------------------------------
            # Result
            if True:
                self.label_result = QLabel('Result: 선택한 아파트와 평가 유사도가 높은 아파트가 추천됩니다.\n '
                                           '- 교통, 교육, 주거환경, 편의시설 수치는 아파트 반경 1km이내 시설 개수와 상대점수입니다.', self.frame1)
                self.label_result.setGeometry(QRect(0, 250, WIDTH, 60))
            #
            self.table_results = QTableWidget(self.frame1)
            self.table_results.setGeometry(QtCore.QRect(0, 310, WIDTH, HEIGHT - 400))
            self.table_results.setRowCount(0)
            #
            header_columns = ['지역구', '아파트', '전용면적', '거래금액', '유사도', '교통', '교육', '주거환경', '편의시설']
            self.table_results.setColumnCount(len(header_columns))
            self.table_results.setHorizontalHeaderLabels(header_columns)
            self.table_results.setEditTriggers(QAbstractItemView.NoEditTriggers)  # edit 금지 모드
            header = self.table_results.horizontalHeader()
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
            #
            header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(6, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(7, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(8, QtWidgets.QHeaderView.ResizeToContents)
            #
            delegate = AlignDelegate(self.table_results)
            for i in range(len(header_columns)):
                self.table_results.setItemDelegateForColumn(i, delegate)  # 중앙정렬
            #
            #
            # ------------------------
            # 최초선택
            # self.user_changed()
            self.user_gu_changed()
            # ------------------------
        except Exception as ex:
            print(str(ex))
    #
    # ==================================================================================================
    def recommends_cf(self):
        print('recommends_cf()')
        try:
            cur_user = self.combo_user.currentText()
            user_info = PD_USERS[PD_USERS.user_id == cur_user]
            print(user_info)
            #
            #
            payload = {
                'user_id': self.combo_user.currentText(),
                'apt_idx': self.combo_apts.currentText()
            }
            response = requests.request("POST", REST_API_URL_CF, data=payload, verify=False)
            #
            # 결과
            res_json = response.json().get('recommends')
            print('res_json:', res_json)
            res_txt = '\n'
            # delete 기존 정보
            for i in reversed(range(self.table_results.rowCount())):
                self.table_results.removeRow(i)
            #
            self.show_table(res_json, filtering=True, user_info=user_info)
            #
        except Exception as ex:
            print(str(ex))
    #
    #
    def recommends_simil(self):
        print('recommends_simil()')
        try:
            if False:
                cur_user = self.combo_user.currentText()
                user_info = PD_USERS[PD_USERS.user_id == cur_user]
                print(user_info)
            #
            #
            payload = {
                'user_id': self.combo_user.currentText(),
                'apt_idx': self.combo_apts.currentText()
            }
            response = requests.request("POST", REST_API_URL_SIMIL, data=payload, verify=False)
            #
            # 결과
            res_json = response.json().get('recommends')
            print('res_json:', res_json)
            #
            # delete 기존 정보
            for i in reversed(range(self.table_results.rowCount())):
                self.table_results.removeRow(i)
            #
            self.show_table(res_json, filtering=False)
            #
        except Exception as ex:
            print(str(ex))
    #
    #
    def show_table(self, res_json, filtering=True, user_info=None):
        try:
            for key1, rcmd_list in res_json.items():
                row_i = 0
                for i in range(len(rcmd_list)):
                    rcmd = rcmd_list[i]
                    apt_idx, corr = rcmd
                    #
                    gu_name = PD_APTS[PD_APTS.apt_idx == apt_idx].get('gu_name').values[0]
                    apt = PD_APTS[PD_APTS.apt_idx == apt_idx].get('apt').values[0]
                    latitude = PD_APTS[PD_APTS.apt_idx == apt_idx].get('latitude').values[0]    # 위도 (apt_y)
                    longitude = PD_APTS[PD_APTS.apt_idx == apt_idx].get('longitude').values[0]  # 경도 (apt_x)
                    area = PD_APTS[PD_APTS.apt_idx == apt_idx].get('area').values[0]
                    price = PD_APTS[PD_APTS.apt_idx == apt_idx].get('price').values[0]
                    print()
                    print(apt, latitude, longitude, area, price)
                    # print(PD_APT_SURR[PD_APT_SURR.apt_addr == apt])
                    pd_rcmd = PD_APT_SURR[PD_APT_SURR.apt_addr == apt][PD_APT_SURR.apt_x == longitude][
                        PD_APT_SURR.apt_y == latitude].drop_duplicates()    # [PD_APT_SURR.apt_area == area]
                    print('pd_rcmd:', pd_rcmd)
                    #
                    cnt_traf = pd_rcmd['교통'].values[0]
                    score_traf = np.round(pd_rcmd['교통_rel'].values[0] * 100, 0)
                    score_traf_total = np.round(pd_rcmd['교통_rel_total'].values[0] * 100, 0)
                    #
                    cnt_edu = pd_rcmd['교육'].values[0]
                    score_edu = np.round(pd_rcmd['교육_rel'].values[0] * 100, 0)
                    score_edu_total = np.round(pd_rcmd['교육_rel_total'].values[0] * 100, 0)
                    #
                    cnt_env = pd_rcmd['주거환경'].values[0]
                    score_env = np.round(pd_rcmd['주거환경_rel'].values[0] * 100, 0)
                    score_env_total = np.round(pd_rcmd['주거환경_rel_total'].values[0] * 100, 0)
                    #
                    cnt_conv = pd_rcmd['편의시설'].values[0]
                    score_conv = np.round(pd_rcmd['편의시설_rel'].values[0] * 100, 0)
                    score_conv_total = np.round(pd_rcmd['편의시설_rel_total'].values[0] * 100, 0)
                    #
                    # --------------------------------------------------------------------------
                    # Filtering
                    if filtering:
                        if price < user_info.get('price_from').values[0] or price > user_info.get('price_to').values[0] * 1.2:
                            continue
                        if area < user_info.get('area_from').values[0] or area > user_info.get('area_to').values[0]:
                            continue
                    # --------------------------------------------------------------------------
                    self.table_results.setRowCount(row_i + 1)
                    self.table_results.setItem(row_i, 0, QTableWidgetItem(gu_name))
                    self.table_results.setItem(row_i, 1, QTableWidgetItem(apt))
                    self.table_results.setItem(row_i, 2, QTableWidgetItem(str(area)))
                    self.table_results.setItem(row_i, 3, QTableWidgetItem(str(price)))
                    self.table_results.setItem(row_i, 4, QTableWidgetItem(str(corr)))
                    #
                    self.table_results.setItem(row_i, 5, QTableWidgetItem('{}개 (구: {}점, 서울시: {}점)'.format(
                        cnt_traf, score_traf, score_traf_total)))
                    self.table_results.setItem(row_i, 6, QTableWidgetItem('{}개 (구: {}점, 서울시: {}점)'.format(
                        cnt_edu, score_edu, score_edu_total)))
                    self.table_results.setItem(row_i, 7, QTableWidgetItem('{}개 (구: {}점, 서울시: {}점)'.format(
                        cnt_env, score_env, score_env_total)))
                    self.table_results.setItem(row_i, 8, QTableWidgetItem('{}개 (구: {}점, 서울시: {}점)'.format(
                        cnt_conv, score_conv, score_conv_total)))
                    row_i += 1
                    #
        except Exception as ex:
            print(str(ex))
    #
    # ==============================================================================================
    def user_gu_changed(self):
        try:
            cur_user_gu = self.combo_user_gu.currentText()
            print('user_gu_changed()', cur_user_gu)
            #
            user_list = PD_USERS[PD_USERS.gu_name==cur_user_gu]['user_id'].drop_duplicates().tolist()
            user_list.sort()
            #
            self.combo_user.clear()
            self.combo_user.addItems(user_list)
            #
        except Exception as ex:
            print(str(ex))
    #
    #
    def user_changed(self):
        try:
            cur_user = self.combo_user.currentText()
            print('user_changed()', cur_user)
            #
            user_info = PD_USERS[PD_USERS.user_id == cur_user]
            user_gender = '남자'
            if user_info.get('gender').values[0] in [2, '2']:
                user_gender = '여자'
            #
            # ---------------------------------------------------
            price_to = user_info.get('price_to').values[0]
            # if price_to > 12:
            #    price_to = ''
            # ---------------------------------------------------
            user_info_str = '이름:{}\n' \
                            '성별:{}\n' \
                            '나이:{}\n' \
                            '희망가격: {}~{}억원\n' \
                            '희망면적: {}~{}m2\n' \
                            '(예상)대출비율: {}'.format(
                user_info.get('name').values[0],
                user_gender,
                user_info.get('age').values[0],
                user_info.get('price_from').values[0], price_to,
                user_info.get('area_from').values[0], user_info.get('area_to').values[0],
                user_info.get('debt_ratio').values[0]
            )
            self.label_user_info.setText(user_info_str)
            #
            # --------------------------------------------------------------
            # 본인이 조회한 아파트 찾기 (eval점수 4점 이상)
            pd_evals = PD_EVALS[PD_EVALS.user_id == cur_user][PD_EVALS.evals >= 4]
            print('pd_evals:', pd_evals)
            #
            apt_list = []
            apt_gu_list = []
            for idx in pd_evals['apt_idx'].tolist():
                apt_list.append(str(idx))
                #
                try:
                    apt_gu = PD_APTS[PD_APTS.apt_idx == idx]['gu_name'].values[0]
                    print(idx, apt_gu)
                    if apt_gu not in apt_gu_list:
                        apt_gu_list.append(apt_gu)
                except Exception as ex:
                    print(str(ex))
            #
            apt_gu_list.sort()
            apt_list.sort()
            #
            #
            if len(apt_list) > 0:
                self.combo_apts.clear()
                self.combo_apts.addItems(apt_list)
            #
            if len(apt_gu_list) > 0:
                self.combo_apts_gu.clear()
                self.combo_apts_gu.addItems(apt_gu_list)
            #
            # --------------------------------------------------------------
        except Exception as ex:
            print(str(ex))
    #
    #
    # ==============================================================================================
    def apt_gu_changed(self):
        try:
            cur_user = self.combo_user.currentText()
            #
            cur_apt_gu = self.combo_apts_gu.currentText()
            print('apt_gu_changed()', cur_user, cur_apt_gu)
            #
            # --------------------------------------------------------------
            # 본인이 조회한 아파트 찾기 (eval점수 4점 이상)
            pd_evals = PD_EVALS[PD_EVALS.user_id == cur_user][PD_EVALS.evals >= 4]
            # print('pd_evals:', pd_evals)
            #
            apt_list = []
            for idx in pd_evals['apt_idx'].tolist():
                try:
                    apt_gu = PD_APTS[PD_APTS.apt_idx == int(idx)]['gu_name'].values[0]
                    if apt_gu == cur_apt_gu:
                        apt_list.append(str(idx))
                except Exception as ex:
                    print(str(ex))
            #
            apt_list.sort()
            #
            if len(apt_list) > 0:
                self.combo_apts.clear()
                self.combo_apts.addItems(apt_list)
            #
            # --------------------------------------------------------------
        except Exception as ex:
            print(str(ex))
    #
    #
    def apt_changed(self):
        try:
            cur_user = self.combo_user.currentText()
            #
            cur_apt = self.combo_apts.currentText()
            print('apt_changed()', cur_user, cur_apt)
            #
            apt_info = PD_APTS[PD_APTS.apt_idx == int(cur_apt)]
            eval_info = PD_EVALS[PD_EVALS.user_id == cur_user][PD_EVALS.apt_idx == int(cur_apt)]
            print('--------------------')
            print(apt_info)
            print('--------------------')
            print(eval_info)
            print('--------------------')
            apt_info_str = '주소:{}\n' \
                            '전용면적:{}\n' \
                            '거래금액:{}\n' \
                            '평점: {}'.format(
                apt_info.get('apt').values[0],
                apt_info.get('area').values[0],
                apt_info.get('price').values[0],
                eval_info.get('evals').values[0]
            )
            self.label_apt_info.setText(apt_info_str)
            #
        except Exception as ex:
            print(str(ex))




# =======================================================================================================
def main():
    app = QApplication(sys.argv)
    win = AptClient()
    print('win_show()')
    win.show()
    print('exit()')
    sys.exit(app.exec_())



if __name__ == '__main__':
    main()



# exe파일 생성용
# pyinstaller --onefile SimpleClient.py --hidden-import='sklearn.utils._cython_blas' --hidden-import='sklearn.neighbors.typedefs' --hidden-import='sklearn.neighbors.quad_tree' --hidden-import='sklearn.tree._utils'

