# -*- coding: utf-8 -*-
import glob
import os
import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from konlpy.tag import Twitter
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns

import pandas as pd
from konlpy.tag import Mecab
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
from adjustText import adjust_text

# ============================================
# 요구사항
# ============================================
# Python을 이용한 역대 대통령의 취임식 연설문에 대한 워드클라우드 이미지 및 막대 그래프


# ============================================
# 보조
# ============================================
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return stopwords


def noun_extractor(text, stopwords):
    tagger = Mecab()
    nouns = tagger.nouns(text)
    # Filter out stopwords and single-character words
    nouns = [noun for noun in nouns if noun not in stopwords and len(noun) > 1]
    return nouns


def create_scatter_plot(data1, data2, keywords1, keywords2):
    # 데이터1과 데이터2의 단어 빈도를 계산합니다.
    counter1 = Counter(data1)
    counter2 = Counter(data2)

    # 데이터1과 데이터2의 단어를 기준으로 키워드를 추출합니다.
    frequencies1 = [counter1[keyword] for keyword in keywords1]
    frequencies2 = [counter2[keyword] for keyword in keywords2]

    print(f'[CHECK] frequencies1 : {frequencies1}')
    print(f'[CHECK] frequencies2 : {frequencies2}')


    saveImg = '{}/{}/{}.png'.format(globalVar['figPath'], serviceName, 'scatter_plot')
    os.makedirs(os.path.dirname(saveImg), exist_ok=True)

    # Scatter Plot을 생성합니다.
    # plt.figure(figsize=(8, 6))
    # plt.figure(figsize=(50, 40))
    # plt.scatter(frequencies1, frequencies2, s=20)

    # plt.title('Scatter Plot', fontsize=150)
    plt.title('Scatter Plot')

    # X축과 Y축의 레이블을 설정합니다.
    # plt.xlabel('기사 Keywords', fontsize=100)
    # plt.ylabel('학술지 Keywords', fontsize=100)
    plt.xlabel('기사 Keywords')
    plt.ylabel('학술지 Keywords')

    print(np.median(frequencies1))

    plt.axvline(x=np.median(frequencies1), linestyle='--', color='red')
    plt.axhline(y=np.median(frequencies2), linestyle='--', color='red')
    # plt.axvline(x=35, linestyle='--', color='red')
    # plt.axhline(y=35, linestyle='--', color='red')

    plt.xlim([0, 70])
    plt.ylim([0, 70])

    textList = []
    # 각 점에 해당하는 키워드를 표시합니다.
    for i, keyword in enumerate(keywords1):
        # plt.text(frequencies1[i], frequencies2[i], keyword,fontproperties=font_path)
        # plt.text(frequencies1[i], frequencies2[i], keyword, fontsize=100)
        # plt.text(frequencies1[i], frequencies2[i], keyword)
        textList.append(plt.text(frequencies1[i], frequencies2[i], keyword))
    adjust_text(textList)

    # plt.axis("off")
    # 그래프를 출력합니다.
    # plt.savefig('scatter_plot.png')
    plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()
    print(f'[CHECK] saveImg : {saveImg}')

# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0438'

# 옵션 설정
sysOpt = {
}

if (platform.system() == 'Windows'):
    globalVar['inpPath'] = './INPUT'
    globalVar['outPath'] = './OUTPUT'
    globalVar['figPath'] = './FIG'
else:
    globalVar['inpPath'] = '/DATA/INPUT'
    globalVar['outPath'] = '/DATA/OUTPUT'
    globalVar['figPath'] = '/DATA/FIG'

# 전역 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# font_path = 'korean_font2.ttf'
# Register the font with Matplotlib
# font_manager.fontManager.addfont(font_path)

# Set the font properties
# font_prop = font_manager.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()


# 데이터 로드 및 전처리
inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'news.csv')
fileList = sorted(glob.glob(inpFile))

# df_article = pd.read_csv('news.csv')  # 기사 데이터가 저장된 CSV 파일 로드
df_article = pd.read_csv(fileList[0])  # 기사 데이터가 저장된 CSV 파일 로드

inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'paper.csv')
fileList = sorted(glob.glob(inpFile))

# df_journal = pd.read_csv('paper.csv')  # 학술지 데이터가 저장된 CSV 파일 로드
df_journal = pd.read_csv(fileList[0])  # 학술지 데이터가 저장된 CSV 파일 로드

inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'stopwords.txt')
fileList = sorted(glob.glob(inpFile))
# stopwords_file = 'stopwords.txt'
stopwords_list = load_stopwords(fileList[0])

# 기사 데이터 전처리
article_texts = df_article['content']  # 본문 데이터 추출
# article_texts = df_article['분석']
article_nouns = []
for text in article_texts:
    nouns = noun_extractor(text, stopwords_list)  # 명사 추출
    article_nouns.extend(nouns)

# 학술지 데이터 전처리
journal_texts = df_journal['Abstract'].astype(str)  # 본문 데이터 추출
journal_nouns = []
for text in journal_texts:
    nouns = noun_extractor(text, stopwords_list)  # 명사 추출
    journal_nouns.extend(nouns)

# 데이터1과 데이터2의 단어를 기준으로 키워드를 추출합니다.
counter1 = Counter(article_nouns)
counter2 = Counter(journal_nouns)
# keywords1 = [item[0] for item in counter1.most_common(20)]  # 상위 20개 키워드 추출
# keywords2 = [item[0] for item in counter2.most_common(20)]  # 상위 20개 키워드 추출

# 상위 100개
keywords1 = [item[0] for item in counter1.most_common(100)]
keywords2 = [item[0] for item in counter2.most_common(100)]

# Scatter Plot 생성
create_scatter_plot(article_nouns, journal_nouns, keywords1, keywords2)

# inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '*.txt')
# fileList = sorted(glob.glob(inpFile))

# fileInfo = fileList[1]
# for i, fileInfo in enumerate(fileList):
#
#     print(f'[CHECK] fileInfo: {fileInfo}')
#
#     fileNameNoExt = os.path.basename(fileInfo).split('.')[0]
#     data = pd.read_csv(fileInfo, delimiter='\t', header=None)
#
#     getData = data[0]
#     getDataTextAll = ' '.join([str(x) for x in getData])
#
#     # 명사만 추출
#     nounList = nlpy.nouns(getDataTextAll)
#
#     # 빈도 계산
#     countList = Counter(nounList)
#
#     dictData = {}
#
#     # 상위 50개 선정
#     for none, cnt in countList.most_common(30):
#         # 빈도수 2 이상
#         if (cnt < 2): continue
#         # 명사  2 글자 이상
#         if (len(none) < 2): continue
#
#         dictData[none] = cnt
#
#     # 빈도분포
#     saveData = pd.DataFrame.from_dict(dictData.items()).rename(
#         {
#             0: 'none'
#             , 1: 'cnt'
#         }
#         , axis=1
#     )
#
#     saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, fileNameNoExt, '빈도분포')
#     os.makedirs(os.path.dirname(saveFile), exist_ok=True)
#     saveData.to_csv(saveFile, index=False)
#     print(f'[CHECK] saveFile : {saveFile}')
#
#     # 빈도분포 시각화
#     saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, fileNameNoExt, '빈도분포')
#     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
#     sns.barplot(x='none', y='cnt', data=saveData)
#     plt.xlabel('명사')
#     plt.ylabel('개수')
#     plt.xticks(rotation=45)
#     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
#     plt.show()
#     plt.close()
#     print(f'[CHECK] saveImg : {saveImg}')
#
#
#     # 워드클라우드
#     wordcloud = WordCloud(
#         width=1000
#         , height=1000
#         , background_color="white"
#         , font_path="NanumGothic.ttf"
#     ).generate_from_frequencies(dictData)
#
#     saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, fileNameNoExt, '워드클라우드')
#     os.makedirs(os.path.dirname(saveImg), exist_ok=True)
#     plt.imshow(wordcloud, interpolation="bilinear")
#     plt.axis("off")
#     plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
#     plt.show()
#     plt.close()
#     print(f'[CHECK] saveImg : {saveImg}')
