# -*- coding: utf-8 -*-
import glob
import os
import platform
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from konlpy.tag import Twitter
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns


# ============================================
# 요구사항
# ============================================
# Python을 이용한 역대 대통령의 취임식 연설문에 대한 워드클라우드 이미지 및 막대 그래프


# ============================================
# 보조
# ============================================

# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0437'

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
nlpy = Twitter()

# 데이터 읽기
inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, '*.txt')
fileList = sorted(glob.glob(inpFile))

fileInfo = fileList[1]
for i, fileInfo in enumerate(fileList):

    print(f'[CHECK] fileInfo: {fileInfo}')

    fileNameNoExt = os.path.basename(fileInfo).split('.')[0]
    data = pd.read_csv(fileInfo, delimiter='\t', header=None)

    getData = data[0]
    getDataTextAll = ' '.join([str(x) for x in getData])

    # 명사만 추출
    nounList = nlpy.nouns(getDataTextAll)

    # 빈도 계산
    countList = Counter(nounList)

    dictData = {}

    # 상위 50개 선정
    for none, cnt in countList.most_common(30):
        # 빈도수 2 이상
        if (cnt < 2): continue
        # 명사  2 글자 이상
        if (len(none) < 2): continue

        dictData[none] = cnt

    # 빈도분포
    saveData = pd.DataFrame.from_dict(dictData.items()).rename(
        {
            0: 'none'
            , 1: 'cnt'
        }
        , axis=1
    )

    saveFile = '{}/{}/{}_{}.csv'.format(globalVar['outPath'], serviceName, fileNameNoExt, '빈도분포')
    os.makedirs(os.path.dirname(saveFile), exist_ok=True)
    saveData.to_csv(saveFile, index=False)
    print(f'[CHECK] saveFile : {saveFile}')

    # 빈도분포 시각화
    saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, fileNameNoExt, '빈도분포')
    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
    sns.barplot(x='none', y='cnt', data=saveData)
    plt.xlabel('명사')
    plt.ylabel('개수')
    plt.xticks(rotation=45)
    plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()
    print(f'[CHECK] saveImg : {saveImg}')


    # 워드클라우드
    wordcloud = WordCloud(
        width=1000
        , height=1000
        , background_color="white"
        , font_path="NanumGothic.ttf"
    ).generate_from_frequencies(dictData)

    saveImg = '{}/{}/{}_{}.png'.format(globalVar['figPath'], serviceName, fileNameNoExt, '워드클라우드')
    os.makedirs(os.path.dirname(saveImg), exist_ok=True)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(saveImg, dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()
    print(f'[CHECK] saveImg : {saveImg}')