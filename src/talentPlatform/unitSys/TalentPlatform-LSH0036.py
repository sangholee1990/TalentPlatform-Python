import logging as log
import os
import sys
import datetime
import dateutil
import pandas as pd
import numpy as np
import email
import imaplib
import configparser
# from plotnine import *
# from plotnine.data import *
# from dfply import *

# 로그 설정
log.basicConfig(stream=sys.stdout, level=log.INFO,
                format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")

contextPath = os.getcwd()

log.info("[START] : %s", "Run Program")

# imap을 이용해서 메일서버에 접근하여 메일 전체를 파일로 다운받고싶은데요(일자별로)
# 다운받은 파일을 db에 넣어서 사용할 예정이라
# 메일내용을 파일로 가져오는것까지 작업이 가능하실까요?
# 특정 계정(이메일)에 대한 메일을 가져와서 파일로 저장하고 싶습니다.
# 예를들면 imap으로 그룹메일에 로그인을 하고 특정계정을 설정하면 해당 계정에 수신된 이메일을 일단위로 텍스트파일로 다운받고 싶습니다.
# 일정은 차주 수요일정두요
# 시작날짜-끝날짜로 예를들면 약 1년치 2019.08.01 ~ 2020.08.01 까지 쌓인 메일을 일별로 쪼개서 가져오고 싶습니다.
# 추후에는 매일 특정시간에 배치로 돌려서 가져오려고 하는데 현재는 과거의 데이터를 가져와서 분석하는게 먼저라서요
# 일별로 데이터가 쌓이면 해당 데이터를 db에 넣어서 사용하려 합니다

# email_address = raw_input("Enter email address: ") if not LOGIN_USERNAME else LOGIN_USERNAME='sangho.lee.1990@gmail.com'
# email_password = getpass("Enter email password: ") if not LOGIN_PASSWORD else LOGIN_PASSWORD


# Email 설정정보 불러오기
config = configparser.ConfigParser()
config['Gmail'] = {}
config['Gmail']['ID'] = '아이디'
config['Gmail']['Password'] = '비밀번호'

# 시작/종료 시간 설정
startDate = "20190801"
endDate = "20200801"

# saveFile = contextPath + '/../resources/data/csv/Gmail_Info_{0}_{1}.csv'.format(startDate, endDate)
saveFile = contextPath + '/../resources/data/xlsx/Gmail_Info_{0}_{1}.xlsx'.format(startDate, endDate)

# gmail imap 세션 생성
session = imaplib.IMAP4_SSL('imap.gmail.com', 993)

# 로그인
session.login(config['Gmail']['ID'], config['Gmail']['Password'])

# 받은편지함
session.select('Inbox')

# 받은 편지함 내 모든 메일 검색
dtStartDate = pd.to_datetime(startDate, format='%Y%m%d').strftime("%d-%b-%Y")
dtEndDate = pd.to_datetime(endDate, format='%Y%m%d').strftime("%d-%b-%Y")

# 특정 날짜 검색
searchOpt = '(SENTSINCE "{0}" SENTBEFORE "{1}")'.format(dtStartDate, dtEndDate)

#  전체 검색
# searchOpt = 'ALL'

log.info("[Check] searchOpt : %s", searchOpt)
result, data = session.search(None, searchOpt)

if (result != 'OK'):
    log.error("[Check] 조회 실패하였습니다.")
    exit(1)

if (len(data) <= 0):
    log.error("[Check] 조회 목록이 없습니다.")
    exit(1)

log.info("[Check] result : %s", result)

# 메일 읽기
emailList = data[0].split()

log.info("[Check] emailList : %s", len(emailList))

# 최근 메일 읽기
# all_email.reverse()

dataL1 = pd.DataFrame()

# for i in range(1, len(emailList)):
for i in range(1, 100):

    mail = emailList[i]
    msgFrom = ''
    msgSender = ''
    msgTo = ''
    msgDate = ''
    subject = ''
    message = ''
    msgDateFmt = ''
    title = ""

    log.info("[Check] mail : %s", mail)

    try:
        result, data = session.fetch(mail, '(RFC822)')
        raw_email = data[0][1]
        raw_email_string = raw_email.decode('utf-8')
        msg = email.message_from_string(raw_email_string)

        # 메일 정보
        msgFrom = msg.get('From')
        msgSender = msg.get('Sender')
        msgTo = msg.get('To')
        msgDate = msg.get('Date')
        msgDateFmt = dateutil.parser.parse(msg.get('Date')).strftime("%Y-%m-%d %H:%M:%S")
        subject = email.header.decode_header(msg.get('Subject'))
        msgEncoding = subject[0][1]
        title = subject[0][0].decode(msgEncoding)

        # for sub in subject:
        #     if isinstance(sub[0], bytes):
        #         title += sub[0].decode(msgEncoding)
        #     else:
        #         title += sub[0]

        if msg.is_multipart():
            for part in msg.get_payload():
                if part.get_content_type() == 'text/plain':
                    bytes = part.get_payload(decode=True)
                    encode = part.get_content_charset()
                    message = message + str(bytes, encode)

                    print, bytes, encode, message
        else:
            if msg.get_content_type() == 'text/plain':
                bytes = msg.get_payload(decode=True)
                encode = msg.get_content_charset()
                message = str(bytes, encode)

        # 딕션너리 정의
        dataInfo = {
            'MailID': [mail]
            , 'From': [msgFrom]
            , 'Sender': [msgSender]
            , 'To': [msgTo]
            , 'Date': [msgDate]
            , 'DateFmt': [msgDateFmt]
            , 'Title': [title]
            , 'Message': [message]
        }

        data = pd.DataFrame(dataInfo)
        dataL1 = data >> bind_rows(dataL1)

    except Exception as e:
        print("Exception : ", e)

session.close()
session.logout()

# dataL1.to_csv(saveFile, sep=',', na_rep='NaN', index=False)
dataL1.to_excel(saveFile)

log.info("[END] : %s", "Run Program")
