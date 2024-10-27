# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import logging.handlers
import os
import platform
import sys
import traceback
import warnings
from _ast import expr
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rc

import seaborn as sns

# =================================================
# 사용자 매뉴얼
# =================================================
# [소스 코드의 실행 순서]
# 1. 초기 설정 : 폰트 설정
# 2. 유틸리티 함수 : 초기화 함수 (로그 설정, 초기 변수, 초기 전달인자 설정) 또는 자주 사용하는 함수
# 3. 주 프로그램 :부 프로그램을 호출
# 4. 부 프로그램 : 자료 처리를 위한 클래스로서 내부 함수 (초기 변수, 비즈니스 로직, 수행 프로그램 설정)
# 4.1. 환경 변수 설정 (로그 설정) : 로그 기록을 위한 설정 정보 읽기
# 4.2. 환경 변수 설정 (초기 변수) : 입력 경로 (inpPath) 및 출력 경로 (outPath) 등을 설정
# 4.3. 초기 변수 (Argument, Option) 설정 : 파이썬 실행 시 전달인자 설정 (pyhton3 *.py argv1 argv2 argv3 ...)
# 4.4. 비즈니스 로직 수행 : 단위 시스템 (unit 파일명)으로 관리 또는 비즈니스 로직 구현

# =================================================
# 1. 초기 설정
# =================================================
warnings.filterwarnings("ignore")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)
# sns.set(font="Malgun Gothic", rc={"axes.unicode_minus": False}, style='darkgrid')

# 그래프에서 마이너스 글꼴 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

# =================================================
# 2. 유틸리티 함수
# =================================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):

    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
        , datetime.now().strftime("%Y%m%d")
    )

    if not os.path.exists(os.path.dirname(saveLogFile)):
        os.makedirs(os.path.dirname(saveLogFile))

    # logger instance 생성
    log = logging.getLogger(prjName)

    if len(log.handlers) > 0:
        return log

    # format 생성
    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    # handler 생성
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(saveLogFile)

    # logger instance에 format 설정
    streamHandler.setFormatter(format)
    fileHandler.setFormatter(format)

    # logger instance에 handler 설정
    log.addHandler(streamHandler)
    log.addHandler(fileHandler)

    # logger instance로 log 기록
    log.setLevel(level=logging.INFO)

    return log


#  초기 변수 설정
def initGlobalVar(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    # 환경 변수 (local, 그 외)에 따라 전역 변수 (입력 자료, 출력 자료 등)를 동적으로 설정
    # 즉 local의 경우 현재 작업 경로 (contextPath)를 기준으로 설정
    # 그 외의 경우 contextPath/resources/input/prjName와 같은 동적으로 구성
    globalVar = {
        'prjName': prjName
        , 'sysOs': platform.system()
        , 'contextPath': contextPath
        , 'resPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources')
        , 'cfgPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config')
        , 'inpPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'input', prjName)
        , 'figPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'fig', prjName)
        , 'outPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'output', prjName)
        , 'movPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'movie', prjName)
        , 'logPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , 'mapPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'mapInfo')
        , 'sysCfg': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'system.json')
        , 'seleniumPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'selenium')
        , 'fontPath': contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'config', 'fontInfo')
    }

    return globalVar


#  초기 전달인자 설정
def initArgument(globalVar, inParams):
    # 원도우 또는 맥 환경
    if globalVar['sysOs'] in 'Windows' or globalVar['sysOs'] in 'Darwin':
        inParInfo = inParams

    # 리눅스 환경
    if globalVar['sysOs'] in 'Linux':
        parser = argparse.ArgumentParser()

        for i, argv in enumerate(sys.argv[1:]):
            if not argv.__contains__('--'): continue
            parser.add_argument(argv)

        inParInfo = vars(parser.parse_args())

        # 글꼴 설정
        # fontList = glob.glob('{}/{}'.format(globalVar['fontPath'], '*.ttf'))
        # fontName = font_manager.FontProperties(fname=fontList[0]).get_name()
        # plt.rcParams['font.family'] = fontName

    log.info(f"[CHECK] inParInfo : {inParInfo}")

    # 전역 변수에 할당
    for key, val in inParInfo.items():
        if val is None: continue
        globalVar[key] = val

    # 전역 변수
    for key, val in globalVar.items():
        if env not in 'local' and key.__contains__('Path') and env and not os.path.exists(val):
            os.makedirs(val)

        globalVar[key] = val.replace('\\', '/')

        log.info(f"[CHECK] {key} : {val}")

    return globalVar

# ================================================
# 4. 부 프로그램
# ================================================
class DtaProcess(object):

    # ================================================
    # 요구사항
    # ================================================
    # Python을 이용한 성별, 연령별에 따른 흡연자 비중 및 비율 시각화

    # ================================================================================================
    # 환경변수 설정
    # ================================================================================================
    global env, contextPath, prjName, serviceName, log, globalVar

    # env = 'local'  # 로컬 : 원도우 환경, 작업환경 (현재 소스 코드 환경 시 .) 설정
    env = 'dev'      # 개발 : 원도우 환경, 작업환경 (사용자 환경 시 contextPath) 설정
    # env = 'oper'  # 운영 : 리눅스 환경, 작업환경 (사용자 환경 시 contextPath) 설정

    if (platform.system() == 'Windows'):
        contextPath = os.getcwd() if env in 'local' else 'E:/04. TalentPlatform/Github/TalentPlatform-Python'
    else:
        contextPath = os.getcwd() if env in 'local' else '/SYSTEMS/PROG/PYTHON/IDE'

    prjName = 'test'
    serviceName = 'LSH0365'

    # 4.1. 환경 변수 설정 (로그 설정)
    log = initLog(env, contextPath, prjName)

    # 4.2. 환경 변수 설정 (초기 변수)
    globalVar = initGlobalVar(env, contextPath, prjName)

    # ================================================================================================
    # 4.3. 초기 변수 (Argument, Option) 설정
    # ================================================================================================
    def __init__(self, inParams):

        log.info('[START] {}'.format("init"))

        try:
            # 초기 전달인자 설정 (파이썬 실행 시)
            # pyhton3 *.py argv1 argv2 argv3 ...
            initArgument(globalVar, inParams)

        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e
        finally:
            log.info('[END] {}'.format("init"))

    # ================================================================================================
    # 4.4. 비즈니스 로직 수행
    # ================================================================================================
    def exec(self):

        log.info('[START] {}'.format("exec"))

        try:

            if (platform.system() == 'Windows'):
                pass
            else:
                globalVar['inpPath'] = '/DATA/INPUT'
                globalVar['outPath'] = '/DATA/OUTPUT'
                globalVar['figPath'] = '/DATA/FIG'

            # 옵션 설정
            sysOpt = {
                # 시작/종료 시간
                'srtDate': '2019-01-01'
                , 'endDate': '2023-01-01'
            }

            import torch
            from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

            # 모델과 토크나이저 로드
            model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
            tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2")

            # 텍스트 생성 함수
            # 텍스트 생성 함수
            def generate_blog_post(prompt, max_length=600):
                input_ids = tokenizer.encode(prompt, return_tensors='pt')

                with torch.no_grad():
                    output = model.generate(input_ids, max_length=max_length, repetition_penalty=2.0,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           bos_token_id=tokenizer.bos_token_id,
                           use_cache=True)

                # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                generated_text = tokenizer.decode(output[0])
                return generated_text

            # 블로그 글 작성 프롬프트
            max_token_limit = model.config.n_positions
            print(f"모델의 최대 토큰 수: {max_token_limit}")

            # prompt = "20~30대 미백을 개선하는 화장품을 소개해조"
            prompt = "맛집 블로그 포스팅을 작성해조."
            prompt = """
            집 블로그 포스팅을 작성할 때는 아래와 같은 요소를 포함하면 좋습니다:

서론 (오프닝)

이 곳을 방문하게 된 이유 (추천받았는지, 우연히 찾았는지 등)
간단한 맛집 소개 (위치나 분위기, 인기 메뉴 등)
간단한 첫인상이나 분위기
맛집 정보

가게 이름과 위치
영업시간과 쉬는 날
주차 정보 (가게 앞, 주변 주차장 등)
주요 메뉴와 추천 메뉴

어떤 메뉴가 인기 있는지, 추천하고 싶은지
메뉴 가격대 및 각 메뉴의 특징
메뉴 선택 이유와 맛 평가
음식 사진과 디테일 설명

음식 사진을 담아내며 각 메뉴의 디테일 설명 (색감, 플레이팅 등)
한 입 먹었을 때의 첫 느낌, 깊은 맛, 질감 등
음식 맛 표현 (매콤, 고소, 신선함, 향신료 등 느낌)
분위기와 인테리어

가게 내부 분위기, 좌석 배치, 인테리어 특징
특이한 소품이나 장식들
가족과 함께 가기 좋은지, 친구와의 모임 장소로 적합한지, 데이트 장소로 좋은지 등
서비스와 직원 응대

직원의 친절함, 서비스 속도
추가적인 서비스 (리필, 서비스 메뉴 제공 등)
총평과 추천 이유

가게 전체적인 장단점 요약
이 곳을 방문해야 할 이유 (특별한 맛, 분위기, 좋은 서비스 등)
다시 가고 싶은지, 혹은 특정한 날에 방문하기 좋은 곳인지
추가 정보

예약 여부, 할인 정보, 주변에 가볼 만한 다른 장소 등
            """
            # 최대 토큰 수에 맞게 조정
            blog_post = generate_blog_post(prompt, max_length=min(max_token_limit, 1024))
            print(blog_post)

            blog_post2 = generate_blog_post('계속', max_length=min(max_token_limit, 1024))
            print(blog_post2)

            # blog_post2 = generate_blog_post(blog_post, max_length=min(max_token_limit, 1000))
            # print(blog_post2)

            # 블로그 글 작성 프롬프트
            # prompt = "오늘의 기술 트렌드에 대해 이야기해보겠습니다. 최근 인공지능의 발전으로"
            # prompt = "2030의 미백을 개선하는 화장품에 대한 광고성 블로그 포스팅을 적어줘. 친절한 언니가 동생에게 하는 어조로 구성"
            # prompt = "2030의 미백을 개선하는 화장품에 대한 광고성 블로그 포스팅을 작성해조."
            # # # prompt = "맛집 블로그 포스팅을 적어줘."
            # blog_post = generate_blog_post(prompt)
            #
            # print(blog_post)
            #
            # def generate_blog_post(prompt, max_length=300):
            #     input_ids = tokenizer.encode(prompt, return_tensors="pt")
            #     output = model.generate(
            #         input_ids,
            #         max_length=max_length,
            #         num_beams=5,
            #         no_repeat_ngram_size=3,
            #         repetition_penalty=1.5,
            #         top_k=50,
            #         top_p=0.95
            #     )
            #     return tokenizer.decode(output[0], skip_special_tokens=True)
            #
            # # 블로그 주제 예시
            # prompt = "2030의 미백을 개선하는 화장품에 대한 광고성 블로그 포스팅"
            #
            # # 텍스트 생성
            # blog_post = generate_blog_post(prompt)
            # print(blog_post)


            # KakaoBrain KoGPT
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            #
            # model_name = "kakaobrain/kogpt"
            # tokenizer = AutoTokenizer.from_pretrained(model_name)
            # model = AutoModelForCausalLM.from_pretrained(model_name)
            #
            # input_text = "한국어 텍스트 생성 예제입니다."
            # input_ids = tokenizer.encode(input_text, return_tensors="pt")
            #
            # output = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True, top_k=50,
            #                         top_p=0.9)
            # generated_text = tokenizer.decode(
            # output[0], skip_special_tokens=True)
            # print(generated_text)


            # Hugging Face Transformers 라
            from transformers import GPT2LMHeadModel, GPT2Tokenizer


            # # 모델과 토크나이저 불러오기
            # model_name = "gpt2"  # 또는 "skt/kogpt2-base-v2" (한국어용 KoGPT2 모델)
            # model = GPT2LMHeadModel.from_pretrained(model_name)
            # tokenizer = GPT2Tokenizer.from_pretrained(model_name)

            # def generate_text(prompt, max_length=50):
            #     # 입력 텍스트를 토큰화
            #     inputs = tokenizer.encode(prompt, return_tensors="pt")
            #     # 텍스트 생성
            #     outputs = model.generate(inputs, max_length=max_length, do_sample=True, pad_token_id=tokenizer.eos_token_id,  top_k=50)
            #     # 생성된 토큰을 텍스트로 변환
            #     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            #     return generated_text



            # # 예제 텍스트 생성
            # prompt = "2030의 미백을 개선하는 화장품에 대한 광고성 블로그 포스팅을 적어줘. 친절한 언니가 동생에게 하는 어조로 구성"
            # # prompt = "맛집 블로그 포스팅을 적어줘."
            # generated_text = generate_text(prompt)
            # print(generated_text)


            # Polyglot-Ko (EleutherAI, 몽고메리)
            # from transformers import AutoTokenizer, AutoModelForCausalLM
            #
            # model_name = "EleutherAI/polyglot-ko-1.3b"
            # tokenizer = AutoTokenizer.from_pretrained(model_name)
            # model = AutoModelForCausalLM.from_pretrained(model_name)
            #
            # input_text = "글쓰기 모델 예제입니다."
            # input_ids = tokenizer.encode(input_text, return_tensors="pt")
            #
            # output = model.generate(input_ids, max_length=50, num_return_sequences=1, do_sample=True, top_k=50,
            #                         top_p=0.9)
            # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # print(generated_text)






            # import json
            # import requests

            # headers = {"Authorization": f"Bearer {API_TOKEN}"}
            # API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
            # def query(payload):
            #     data = json.dumps(payload)
            #     response = requests.request("POST", API_URL, headers=headers, data=data)
            #     return json.loads(response.content.decode("utf-8"))
            # data = query(
            #     {
            #         "inputs": {
            #             "question": "기후변환 블로그 글써조",
            #             "context": "기후변환 블로그 글써조",
            #         }
            #     }
            # )
            #
            # print(data)

            #
            # data = {
            #     # "inputs": "Can you please let us know more details about your "
            #     "inputs": "맛집 블로그 글 써조"
            # }
            # response = requests.post(API_URL, headers=headers, json=data)
            # print(response.json())

            # def query(payload):
            #     data = json.dumps(payload)
            #     response = requests.request("POST", API_URL, headers=headers, data=data)
            #     return json.loads(response.content.decode("utf-8"))
            #
            # data = query({"inputs": "맛집 블로그 글 써조"})
            # print(data)



            # inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'HN_M.csv')
            # fileList = sorted(glob.glob(inpFile))

            # import torch
            # from transformers import AutoTokenizer, AutoModelForCausalLM
            #
            # tokenizer = AutoTokenizer.from_pretrained(
            #     'kakaobrain/kogpt',  # KoGPT 모델의 이름 지정
            #     revision='KoGPT6B-ryan1.5b-float16',  # 모델 버전 선택
            #     bos_token='[BOS]',  # 문장의 시작을 나타내는 토큰
            #     eos_token='[EOS]',  # 문장의 끝을 나타내는 토큰
            #     unk_token='[UNK]',  # 미지의 단어를 나타내는 토큰
            #     pad_token='[PAD]',  # 패딩 토큰
            #     mask_token='[MASK]'  # 마스크 토큰
            # )
            #
            # model = AutoModelForCausalLM.from_pretrained(
            #     'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',
            #     # or float32 version: revision=KoGPT6B-ryan1.5b
            #     pad_token_id=tokenizer.eos_token_id,
            #     torch_dtype='auto', low_cpu_mem_usage=True
            # ).to(device='cuda', non_blocking=True)
            # _ = model.eval()
            #
            # prompt = '인간처럼 생각하고, 행동하는 \'지능\'을 통해 인류가 이제까지 풀지 못했던'
            # with torch.no_grad():
            #     tokens = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
            #     gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=64)
            #     generated = tokenizer.batch_decode(gen_tokens)[0]
            #
            # print(generated)


            from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

            import torch
            from transformers import GPT2LMHeadModel

            model_name = "skt/kogpt2-base-v2"
            model = GPT2LMHeadModel.from_pretrained(model_name)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                model_name,
                bos_token='</s>',
                eos_token='</s>',
                unk_token='<unk>',
                pad_token='<pad>'
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            def generate_complete_text(prompt, max_new_tokens=150, max_length=1000):
                generated_text = prompt
                while len(generated_text) < max_length:
                    new_text = generate_korean_text2(generated_text, max_new_tokens=max_new_tokens)

                    # 중복된 텍스트 제거
                    generated_text += new_text[len(generated_text):]

                    # 텍스트가 eos_token으로 끝났다면 종료
                    if new_text.endswith(tokenizer.eos_token):
                        break

                return generated_text

            def generate_korean_text2(prompt, max_new_tokens=150):
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    # do_sample=True,
                    # top_p=0.95,
                    # top_k=50,
                    # # top_p=0.90,
                    # # top_k=40,
                    # pad_token_id=tokenizer.pad_token_id,
                    # eos_token_id=tokenizer.eos_token_id,
                    # early_stopping=True,
                    # # early_stopping=False,
                    # no_repeat_ngram_size=2,
                    # repetition_penalty=1.2,
                    # temperature=0.8
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                return generated_text

            # prompt = """2030의 미백을 개선하는 화장품에 대한 광고성 블로그 포스팅을 적어줘. 친절한 언니가 동생에게 하는 어조로 구성"""
            # prompt = "2030의 미백을 개선하는 화장품에 대한 광고성 블로그 포스팅을 적어줘. 친절한 언니가 동생에게 하는 어조로 구성"
            prompt = "맛집 블로그 포스팅을 적어줘."

            # prompt = "2024년 디지털 마케팅 트렌드는"
            # blog_post = generate_korean_text2(prompt, max_new_tokens=500)
            blog_post = generate_complete_text(prompt, max_new_tokens=150, max_length=2000)
            print(blog_post)

                # # 문장 단위로 분할
                # import re
                # sentences = re.split('(?<=[.!?])\s', generated_text)
                # total_tokens = 0
                # final_text = ''
                #
                # for sentence in sentences:
                #     token_count = len(tokenizer.encode(sentence))
                #     if total_tokens + token_count <= max_new_tokens:
                #         final_text += sentence + ' '
                #         total_tokens += token_count
                #     else:
                #         break
                # return final_text.strip()

            def generate_long_korean_text(prompt, total_characters=2000, max_new_tokens=150):
                generated_text = prompt
                while True:
                    inputs = tokenizer.encode(generated_text, return_tensors="pt").to(device)
                    input_length = inputs.shape[1]

                    if input_length > 1024:
                        inputs = inputs[:, -1024:]
                        input_length = inputs.shape[1]

                    max_allowed_new_tokens = 1024 - input_length
                    current_max_new_tokens = min(max_new_tokens, max_allowed_new_tokens)

                    outputs = model.generate(
                        inputs,
                        max_new_tokens=current_max_new_tokens,
                        do_sample=True,
                        top_p=0.95,
                        top_k=50,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        early_stopping=True,
                        no_repeat_ngram_size=2,
                        repetition_penalty=1.2,
                        # temperature=0.8
                    )

                    new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    new_generated_text = new_text[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):]
                    generated_text += new_generated_text

                    if len(generated_text) >= total_characters:
                        generated_text = generated_text[:total_characters]
                        break

                return generated_text

            # def generate_korean_text2(prompt, max_new_tokens=150, max_total_length=1000):
            #     # 초기 입력을 시작 텍스트로 설정
            #     current_text = prompt
            #
            #     while len(current_text) < max_total_length:
            #         inputs = tokenizer.encode(current_text, return_tensors="pt")
            #
            #         # 새로운 텍스트 생성
            #         outputs = model.generate(
            #             inputs,
            #             max_new_tokens=max_new_tokens,
            #             do_sample=True,
            #             top_p=0.95,
            #             top_k=50,
            #             pad_token_id=tokenizer.pad_token_id,
            #             no_repeat_ngram_size=2,
            #             repetition_penalty=1.2,
            #             temperature=0.8,
            #             eos_token_id=tokenizer.eos_token_id,
            #             early_stopping=True
            #         )
            #
            #         # 생성된 텍스트를 디코딩하고 현재 텍스트에 추가
            #         new_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            #
            #         # 이전에 사용된 텍스트 부분 제거
            #         if new_text.startswith(current_text):
            #             new_text = new_text[len(current_text):]
            #
            #         # 새로운 텍스트를 현재 텍스트에 추가
            #         current_text += new_text
            #
            #         # 종료 조건 확인: EOS 토큰이 있거나 최대 길이에 도달한 경우
            #         if tokenizer.eos_token_id in outputs[0] or len(current_text) >= max_total_length:
            #             # 글이 마무리될 때 추가적인 마무리 문장을 삽입합니다.
            #             current_text += "\n\n감사합니다. 이로써 이야기가 끝났습니다."
            #             break
            #
            #     return current_text

            #
            def generate_korean_text(prompt, max_new_tokens=150):
                inputs = tokenizer.encode(prompt, return_tensors="pt")
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.2,
                    temperature=0.8,
                    eos_token_id=tokenizer.eos_token_id,
                    # eos_token_id=None,
                    early_stopping=True
                )
                return tokenizer.decode(outputs[0], skip_special_tokens=True)
            #
            # def generate_korean_text2(prompt, max_new_tokens=150):
            #     inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            #
            #     # 모델의 최대 길이 확인 (예: 1024)
            #     max_model_length = 1024
            #     input_length = inputs.shape[1]
            #
            #     # 입력 길이가 모델의 최대 길이를 초과하지 않도록 조정
            #     if input_length >= max_model_length:
            #         inputs = inputs[:, -max_model_length + max_new_tokens:]
            #         input_length = inputs.shape[1]
            #
            #     outputs = model.generate(
            #         inputs,
            #         max_new_tokens=max_new_tokens,
            #         do_sample=True,
            #         top_p=0.95,
            #         top_k=50,
            #         pad_token_id=tokenizer.pad_token_id,
            #         no_repeat_ngram_size=2,
            #         repetition_penalty=1.2,
            #         temperature=0.8,
            #         eos_token_id=None,
            #         early_stopping=False
            #     )
            #     return tokenizer.decode(outputs[0], skip_special_tokens=True)

            # prompt = "기후 변화는 지구 환경에 큰 영향을 미치고 있습니다."
            prompt = """
            인기있는 맛집 포스팅 해줘
            """
            # generated_text = generate_korean_text(prompt)
            # print(generate_korean_text(prompt, max_new_tokens=100))
            # print(generate_korean_text2(prompt, max_new_tokens=1000))
            #
            # aa = generate_long_korean_text(prompt, total_characters=1000, max_new_tokens=150)
            # print(aa)
            # print(len(aa))

            # aa = generate_korean_text(prompt, max_new_tokens=5000)
            # # aa = generate_korean_text2(prompt, max_new_tokens=150, max_total_length=1000)
            # aa = generate_korean_text(prompt, max_new_tokens=150)
            # print(aa)
            # print(len(aa))
            #
            # conversation = prompt
            # response = generate_korean_text(conversation)
            # conversation += response
            # print(response)
            # print(len(response))

            conversation = """2030의 미백을 개선하는 화장품에 대한 광고성 블로그 포스팅을 적어줘. 친절한 언니가 동생에게 하는 어조로 구성"""
            # responseAll = ""
            response = generate_korean_text(conversation, max_new_tokens=800)
            print(response, len(response))
            # responseAll += response






            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            # 모델과 토크나이저 로드
            tokenizer = GPT2Tokenizer.from_pretrained("skt/kogpt2-base-v2")
            model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")

            # model_name = "skt/kogpt3-base"  # 이 모델 이름은 확인이 필요합니다
            # model = GPT2LMHeadModel.from_pretrained(model_name)
            # tokenizer = GPT2Tokenizer.from_pretrained(model_name)

            # 텍스트 생성
            # text = """2030의 미백을 개선하는 화장품에 대한 블로그 포스팅을 적어줘. 친절한 언니가 동생에게 하는 어조로 구성"""
            text = """2030의 미백을 개선하는 화장품에 대한 블로그 포스팅을 적어줘"""
            input_ids = tokenizer.encode(text, return_tensors='pt')
            output = model.generate(input_ids, max_length=200)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(generated_text)

            from transformers import BertForSequenceClassification, BertTokenizer
            from torch.nn.functional import softmax
            import torch

            model_name = "monologg/kobert"  # Hugging Face에 등록된 KoBERT 모델
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 예: 이진 분류
            tokenizer = BertTokenizer.from_pretrained(model_name)

            # text = "이 제품 정말 좋네요!"
            text = """2030의 미백을 개선하는 화장품에 대한 블로그 포스팅을 적어줘"""
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = softmax(outputs.logits, dim=-1)
            print("Review sentiment:", "Positive" if predictions[0][1] > 0.5 else "Negative")

            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            model_name = "gpt2"  # 기본 GPT-2 모델
            model = GPT2LMHeadModel.from_pretrained(model_name)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)

            # 블로그 포스트의 주제 예: "여행 팁"
            # prompt = "여행을 떠나기 전 알아야 할 최고의 팁:"
            prompt = """2030의 미백을 개선하는 화장품에 대한 블로그 포스팅을 적어줘"""
            inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

            # 텍스트 생성
            outputs = model.generate(inputs, max_length=500, num_return_sequences=1)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            print("Generated Blog Post Draft:\n", generated_text)

            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            def generate_blog_post(topic, max_length=300):
                # GPT-2 모델과 토크나이저 로드 (skt/kogpt2-base-v2 같은 한국어 모델도 사용할 수 있음)
                model_name = "gpt2"  # 한국어 모델: "skt/kogpt2-base-v2"
                model = GPT2LMHeadModel.from_pretrained(model_name)
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)

                # 입력된 주제를 기반으로 포스팅 초안 작성
                prompt = f"블로그 포스팅 주제: {topic}\n"
                inputs = tokenizer.encode(prompt, return_tensors="pt")

                # GPT-2 모델을 이용한 텍스트 생성
                outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2,
                                         early_stopping=True)

                # 결과 텍스트 디코딩
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                return generated_text

            # 주제 입력 및 포스팅 생성
            # topic = input("블로그 포스팅 주제를 입력하세요: ")
            topic = input("""2030의 미백을 개선하는 화장품에 대한 블로그 포스팅을 적어줘""")
            blog_post = generate_blog_post(topic)
            # print("\n=== 생성된 블로그 포스팅 초안 ===\n")
            print(blog_post)
            #
            # response = generate_korean_text(responseAll)
            # print(response, len(response))
            # responseAll += response

            # generate_korean_text(prompt, max_new_tokens=150)
            # print(generate_korean_text(prompt, max_length=1024))




            # prompt = """https://blog.naver.com/dnjsfudcjs/223594977298

            # 맛집 포스팅 200자 이내로 해조"""
            # generated_text = generate_korean_text2(prompt, max_new_tokens=200)
            # print(generated_text)
            # len(aa)


        except Exception as e:
            log.error(f"Exception : {str(e)}")
            raise e
        finally:
            log.info('[END] {}'.format("exec"))

# ================================================
# 3. 주 프로그램
# ================================================
if __name__ == '__main__':

    print('[START] {}'.format("main"))

    try:

        # 파이썬 실행 시 전달인자를 초기 환경변수 설정
        inParams = { }
        print("[CHECK] inParams : {}".format(inParams))

        # 부 프로그램 호출
        subDtaProcess = DtaProcess(inParams)

        subDtaProcess.exec()

    except Exception as e:
        print(traceback.format_exc())
        sys.exit(1)

    finally:
        print('[END] {}'.format("main"))