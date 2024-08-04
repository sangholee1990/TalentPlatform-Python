# %%
from pororo import Pororo
import pandas as pd
from datetime import datetime

# %%
class SentimentAnalysis:
    def __init__(self):
        self.nowdate = datetime.today().strftime("%y%m%d")

        pass

    def readData(self, save_path, crawling_site, product, part):
        
        try:
            temp_df = pd.read_csv(f'{save_path}/REVIEW_RAW_{crawling_site}_{product}_{self.nowdate}.csv',
                                    encoding='utf-8')
        except UnicodeDecodeError:
            temp_df = pd.read_csv(f'{save_path}/REVIEW_RAW_{crawling_site}_{product}_{self.nowdate}.csv',
                                    encoding='cp949')
        '''
        if part == "1":
            start_idx = 0
            end_idx = (len(temp_df)//5) * int(part)
        elif part == "5":
            start_idx = (len(temp_df)//5) * (int(part) - 1)
            end_idx = len(temp_df)
        else:
            start_idx = (len(temp_df)//5) * (int(part) - 1)
            end_idx = (len(temp_df)//5) * int(part)
        '''
        '''
        if int(start_idx) > len(temp_df):
            raise Exception("Excess Maxium Dataframe Index")
        if int(end_idx) > len(temp_df):
            end_idx = len(temp_df)
        '''
        
        # self.data = temp_df.loc[range(int(start_idx), int(end_idx))]

        self.data = temp_df

        self.site = crawling_site
        self.product = product

        # print(self.data.head())
        # print(self.data.tail())
        
    def usePororo(self, category_list, save_path, part):
        # 참조 평가점수
        refReview = Pororo(task="review", lang="ko")

        # 참조 감정분석
        refSentiment = Pororo(task='sentiment', model = 'brainbert.base.ko.shopping', lang='ko')

        # 참조 주제 분류
        refTopic = Pororo(task="zero-topic")

        # 참조 요약
        refSumm = Pororo(task="summarization", model="abstractive", lang="ko")

        categoryList = category_list.split(",")
        # print(categoryList)

        for i, row in self.data.iterrows():
            revierContType = '요약'

            # 리뷰 본문의 글자수 512 초과 시 요약문 대체
            reviewCont = refSumm(row.content)

            try:
                reviewSentimentInfo = refSentiment(row.content, show_probs=True)
                reviewTopicInfo = refTopic(row.content, categoryList)
                reviewCont = row.content
                revierContType = '본문'
            except Exception as e:
                print(e)
                
            reviewGrade = None
            reviewSentimentInfo = None
            reviewSentimentPos = None
            reviewSentimentNeg = None
            reviewSentimentBest = None
            reviewTopicInfo = None
            reviewTopic = None
            reviewTopic2 = None
            reviewTopic3 = None
            reviewTopic4 = None
            reviewTopic5 = None
            reviewTopicBest = None
            reviewSumm = None

            try:
                # 요약
                reviewSumm = refSumm(row.content)

                # 평가점수
                reviewGrade = refReview(reviewCont)

                # 감정분석
                reviewSentimentInfo = refSentiment(reviewCont, show_probs=True)
                reviewSentimentPos = reviewSentimentInfo.get('positive') * 100.0
                reviewSentimentNeg = reviewSentimentInfo.get('negative') * 100.0
                reviewSentimentBest = '긍정' if reviewSentimentPos > reviewSentimentNeg else '부정'

                # 주제 분류
                reviewTopicInfo = refTopic(reviewCont, categoryList)
                reviewTopic = reviewTopicInfo.get(categoryList[0])
                reviewTopic2 = reviewTopicInfo.get(categoryList[1])
                reviewTopic3 = reviewTopicInfo.get(categoryList[2])
                reviewTopicBest = max(reviewTopicInfo.keys(), key=(lambda i: reviewTopicInfo[i]))
                reviewTopic4 = reviewTopicInfo.get(categoryList[3])
                reviewTopic5 = reviewTopicInfo.get(categoryList[4])

            except Exception as e:
                print(e)
        #         log.error("Exception : {}".format(e))

            self.data.loc[i, 'revierContType'] = revierContType
            self.data.loc[i, 'reviewSumm'] = reviewSumm
            self.data.loc[i, 'reviewGrade'] = reviewGrade
            self.data.loc[i, 'reviewSentimentPos'] = reviewSentimentPos
            self.data.loc[i, 'reviewSentimentNeg'] = reviewSentimentNeg
            self.data.loc[i, 'reviewSentimentBest'] = reviewSentimentBest
            self.data.loc[i, 'reviewTopic'] = reviewTopic
            self.data.loc[i, 'reviewTopic2'] = reviewTopic2
            self.data.loc[i, 'reviewTopic3'] = reviewTopic3
            self.data.loc[i, 'reviewTopic4'] = reviewTopic4
            self.data.loc[i, 'reviewTopic5'] = reviewTopic5
            self.data.loc[i, 'reviewTopicBest'] = reviewTopicBest
            
            if i != 0 and i % 20 == 0:
                self.data.to_csv(f'{save_path}/REVIEW_SENTI_{self.site}_{self.product}_PART{part}_{self.nowdate}.csv', index=False)

    def saveData(self, save_path, part):
        self.data.to_csv(f'{save_path}/REVIEW_SENTI_{self.site}_{self.product}_PART{part}_{self.nowdate}.csv', index=False)

        print(f"DATA_SAVE : {self.product}")