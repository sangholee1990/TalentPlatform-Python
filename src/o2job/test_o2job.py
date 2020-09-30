# imap을 이용해서 메일서버에 접근하여 메일 전체를 파일로 다운받고싶은데요(일자별로)
# 다운받은 파일을 db에 넣어서 사용할 예정이라
# 메일내용을 파일로 가져오는것까지 작업이 가능하실까요?
# 특정 계정(이메일)에 대한 메일을 가져와서 파일로 저장하고 싶습니다.
# 예를들면 imap으로 그룹메일에 로그인을 하고 특정계정을 설정하면 해당 계정에 수신된 이메일을 일단위로 텍스트파일로 다운받고 싶습니다.
# 일정은 차주 수요일정두요
# 시작날짜-끝날짜로 예를들면 약 1년치 2019.08.01 ~ 2020.08.01 까지 쌓인 메일을 일별로 쪼개서 가져오고 싶습니다.
# 추후에는 매일 특정시간에 배치로 돌려서 가져오려고 하는데 현재는 과거의 데이터를 가져와서 분석하는게 먼저라서요
# 일별로 데이터가 쌓이면 해당 데이터를 db에 넣어서 사용하려 합니다

# 라이브러리 읽기
import logging as log
import os
import sys
import dateutil
import pandas as pd
import email
import imaplib
import configparser
import matplotlib as mpl
import matplotlib.pyplot as plt
import traceback
from email.header import decode_header
from datetime import datetime

# 로그 설정
log.basicConfig(stream=sys.stdout, level=log.INFO,
                format="%(asctime)s [%(name)s | %(lineno)d | %(filename)s | %(funcName)10.10s] [%(levelname)-5.5s] %(message)s")
# warnings.filterwarnings("ignore")
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# 그래프에서 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

try:
    log.info('[START] main : {}'.format('Run Program'))

    # 작업환경 경로 설정
    contextPath = os.getcwd()

    # 전역 변수
    globalVar = {
        "contextPath": {
            "img": contextPath + '/../../resources/image/'
            , "csv": contextPath + '/../../resources/data/csv/'
            , "xlsx": contextPath + '/../../resources/data/xlsx/'
        }
        , "config": {
            "system": contextPath + '/../../resources/config/system.cfg'
        }
    }

    log.info("[Check] globalVar : {}".format(globalVar))

    # ==============================================
    # 주 소스코드
    # ==============================================
    log.info("[START] Main : %s", "Run Program")

    # 메일 및 시작/종료 시간 설정
    # Gmail 정보
    # mailType = 'gmail'

    # Naver 정보
    # mailType = 'naver'

    # Line Work 정보
    mailType = 'lineWorks'

    # 시작/종료 시간 설정
    startDate = "20180101"
    endDate = "20200801"

    # Email 설정정보 불러오기
    systemConfigName = globalVar.get('config').get('system')

    config = configparser.ConfigParser()
    config.read(systemConfigName, encoding='utf-8')

    if mailType.__contains__('gmail'):
        imap4Info = 'imap.gmail.com'
    elif mailType.__contains__('naver'):
        imap4Info = 'imap.naver.com'
    elif mailType.__contains__('lineWorks'):
        imap4Info = 'imap.worksmobile.com'
    else:
        mailType = 'gmail'
        imap4Info = 'imap.gmail.com'

    # 받은 편지함 내 모든 메일 검색
    dtStartDate = pd.to_datetime(startDate, format='%Y%m%d')
    dtEndDate = pd.to_datetime(endDate, format='%Y%m%d')

    sStartDate = dtStartDate.strftime("%d-%b-%Y")
    sEndDate = dtEndDate.strftime("%d-%b-%Y")

    saveFile = globalVar.get('contextPath').get('xlsx') + '{}_Info_{}_{}.xlsx'.format(mailType, startDate, endDate)

    # ==================================================
    # 메일 로그인
    # ==================================================
    id = config.get(mailType, 'id')
    password = config.get(mailType, 'password')

    session = imaplib.IMAP4_SSL(imap4Info, 993)
    session.login(id, password)

    # 받은편지함
    session.select('Inbox')

    if mailType.__contains__('gmail'):
        # 특정 날짜 검색
        searchOpt = '(SENTSINCE "{0}" SENTBEFORE "{1}")'.format(sStartDate, sEndDate)
    else:
        #  전체 검색
        searchOpt = 'ALL'

    log.info("[Check] searchOpt : %s", searchOpt)
    result, data = session.search(None, searchOpt)

    if (result != 'OK'):
        log.error("[Check] 조회 실패하였습니다.")

    log.info("[Check] result : %s", result)

    # 메일 읽기
    emailList = data[0].split()

    if (len(emailList) <= 0):
        log.error("[Check] 조회 목록이 없습니다.")

    log.info("[Check] emailList : %s", len(emailList))

    dataL1 = pd.DataFrame()

    # i = 1000
    # i = 1
    # for i in range(1, len(emailList)):
    for i in range(1, 100):

        msgFrom = ''
        msgSender = ''
        msgTo = ''
        msgDate = ''
        subject = ''
        message = ''
        msgDateFmt = ''
        title = ""

        try:
            mail = emailList[i]
            log.info("[Check] mail : %s", mail)

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

            dtDateUnix = datetime.timestamp(pd.to_datetime(msgDate))
            dtStartDateUnix = datetime.timestamp(dtStartDate)
            dtEndDateUnix = datetime.timestamp(dtEndDate)

            if (dtDateUnix < dtStartDateUnix) or (dtEndDateUnix < dtDateUnix):
                continue

            if msg.is_multipart():
                for part in msg.get_payload():
                    if part.get_content_type() == 'text/plain':
                        bytes = part.get_payload(decode=True)
                        encode = part.get_content_charset()
                        if (encode == None): encode = 'UTF-8'
                        message = message + str(bytes, encode)

            else:
                if msg.get_content_type() == 'text/plain':
                    bytes = msg.get_payload(decode=True)
                    encode = msg.get_content_charset()
                    if (encode == None): encode = 'UTF-8'
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
            dataL1 = dataL1.append(data)

        except Exception as e:
            log.error("Exception : {}".format(e))

    session.close()
    session.logout()

    dataL1.to_excel(saveFile)

except Exception as e:
    log.error("Exception : {}".format(e))
    # traceback.print_exc()
    # sys.exit(1)

finally:
    log.info('[END] Main : {}'.format('Run Program'))