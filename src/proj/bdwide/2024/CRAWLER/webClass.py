# %%
# 기본 라이브러리
import bs4
import pandas as pd
import numpy as np
import re
import time
import requests
import math
import os
from datetime import datetime

# 데이터 수집을 위한 selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from pathlib import Path

import chromedriver_autoinstaller

# %%
# 기본 함수. 스크롤 내리기
def scroll_down(in_driver):
    #스크롤 내리기 이동 전 위치 
    scroll_location = in_driver.execute_script("return document.body.scrollHeight") 

    while True: 
        #현재 스크롤의 가장 아래로 내림 
        in_driver.execute_script("window.scrollTo(0,document.body.scrollHeight)") 

        #전체 스크롤이 늘어날 때까지 대기 
        time.sleep(2) 

        #늘어난 스크롤 높이 
        scroll_height = in_driver.execute_script("return document.body.scrollHeight") 

        #늘어난 스크롤 위치와 이동 전 위치 같으면(더 이상 스크롤이 늘어나지 않으면) 종료 
        if scroll_location == scroll_height: 
            break 

        #같지 않으면 스크롤 위치 값을 수정하여 같아질 때까지 반복 
        else: 
            #스크롤 위치값을 수정 
            scroll_location = in_driver.execute_script("return document.body.scrollHeight")

# %%
# 기본 함수. 상품 페이지 닫기
def page_close(in_driver):
    in_driver.close()

    time.sleep(1)

    in_driver.switch_to.window(in_driver.window_handles[0])


# %%
# 데이터 수집용 WebCrawler 클래스 선언
class WebCrawler:
    def __init__(self):
        # self.driver_path = '/usr/bin/chromedriver'
        
        # 22.01.01 형식
        self.nowdate = datetime.today().strftime("%y%m%d")
        
        # 2022/01/01 형식
        self.nowdate2 = datetime.today().strftime("%Y/%m/%d")
        
        # 상품과 관련된 정보
        self.goods_df = pd.DataFrame()
        
        # 리뷰와 관련된 정보
        self.review_df = pd.DataFrame()
        pass
    
    # 셀레니움 드라이버 활성화
    def open_driver(self):
        driver_path = '/usr/bin/chromedriver'
        option = Options()
        # option.add_argument("disable-gpu")
        # option.add_argument('window-size=1920x1080')
        # option.add_argument("lang=ko_KR")
        option.add_argument('--headless')
        option.add_argument('--no-sandbox')
        option.add_argument('--disable-dev-shm-usage')

        # 2023.07.17 기준 Chrome 114 버전
        driver = webdriver.Chrome(driver_path, options=option)

        '''
        chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0]
        try:
            driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver', options=option)
        except:
            chromedriver_autoinstaller.install(True)
            driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver', options=option)
        '''

        time.sleep(3)
        
        self.driver = driver
                
    # 사이트 메인 페이지로 이동
    def move_mainPage(self, site):
        self.search_site = site
        
        if site == "gmarket":
            url = f'https://www.gmarket.co.kr/'
        elif site == "naver":
            url = f'https://shopping.naver.com/home/p/index.naver'
        elif site == "ssg":
            url = f'https://www.ssg.com/'
            
        self.driver.get(url)

        time.sleep(5)        
        
        # 2022.01.01 형식
        self.nowdate3 = datetime.today().strftime("%Y.%m.%d")

    # 상품 목록 페이지로 이동
    def move_listPage(self, product):
        # 각 사이트의 메인 화면에서 특정 제품에 대한 값을 입력하여 검색하는 방법 사용
        if self.search_site == "gmarket":
            self.driver.find_element(by=By.XPATH, 
                                     value='//*[@id="skip-navigation-search"]/span/input').send_keys(product)
            self.driver.find_element(by=By.XPATH, 
                                     value='//*[@id="skip-navigation-search"]/span/input').send_keys(Keys.RETURN)
        elif self.search_site == "naver":
            # 네이버의 경우 검색 형태가 자주 바껴서 확인이 필요함
            try:
                self.driver.find_element(by=By.XPATH, value='//*[@id="autocompleteWrapper"]/input[1]').send_keys(product)
                self.driver.find_element(by=By.XPATH, value='//*[@id="autocompleteWrapper"]/a[2]').send_keys(Keys.RETURN)
            except NoSuchElementException:
                try:
                    self.driver.find_element(by=By.XPATH, 
                                            value='//*[@id="__next"]/div/div[1]/div/div/div[2]/div/div[2]/div/div[2]/form/fieldset/div[1]/div/input').send_keys(product)
                    self.driver.find_element(by=By.XPATH, 
                                            value='//*[@id="__next"]/div/div[1]/div/div/div[2]/div/div[2]/div/div[2]/form/fieldset/div[1]/div/button[2]').send_keys(Keys.RETURN)
                except NoSuchElementException:
                    self.driver.find_element(by=By.XPATH, 
                                            value='//*[@id="__next"]/div/div[1]/div/div/div[2]/div/div[2]/div/div[2]/form/div[1]/div[1]/input').send_keys(product)
                    self.driver.find_element(by=By.XPATH, 
                                            value='//*[@id="__next"]/div/div[1]/div/div/div[2]/div/div[2]/div/div[2]/form/div[1]/div/button[2]').send_keys(Keys.RETURN)
        elif self.search_site == "ssg":
            self.driver.find_element(by=By.XPATH, value='//*[@id="ssg-query"]').click()
            self.driver.find_element(by=By.XPATH, value='//*[@id="ssg-query"]').send_keys(product)
            self.driver.find_element(by=By.XPATH, value='//*[@id="ssg-query"]').send_keys(Keys.RETURN)
                    
        self.driver.implicitly_wait(5)

        time.sleep(5)

        print(f"{self.search_site} MOVE_PAGE : {product}")
        
        # 지마켓의 경우 상품의 순서를 G마켓 랭킹이 아닌 판매 인기 순으로 변경
        if self.search_site == "gmarket":
            self.driver.find_element(by=By.CLASS_NAME, value="button__toggle-sort").click()
            self.driver.implicitly_wait(1)
            
            self.driver.find_element(by=By.XPATH, value='//*[@class="box__sort-control-list"]/ul/li[2]/a').click()
            time.sleep(3)
        elif self.search_site == "ssg":
            self.driver.find_element(by=By.XPATH, value='//*[@id="content"]/div[7]/div[1]/div/div[2]/div/ul/li[1]/div/div[1]/a').click()
            self.driver.implicitly_wait(1)

            self.driver.find_element(by=By.XPATH, value='//*[@id="content"]/div[7]/div[1]/div/div[2]/div/ul/li[1]/div/div[1]/div/ul/li[2]/a').click()
        
        scroll_down(self.driver)
    
    # 필요한 상품 목록 추출
    def index_list(self):
        in_url_list = list()

        response = self.driver.page_source

        bs_obj = bs4.BeautifulSoup(response, 'html.parser')
        if self.search_site == "gmarket":            
            box_list = bs_obj.select("#section__inner-content-body-container>div>div.box__component")
            
            for box in box_list:
                box_a = box.find('a', class_='link__item')
                try:
                    product_url = box_a["href"]
                    in_url_list.append(product_url)
                except:
                    pass
        elif self.search_site == "naver":
            print(self.driver.current_url)
            box_list = bs_obj.select("ul.list_basis>div>div")
            if len(box_list) == 0:
                print("BOX ERROR")
                box_list = bs_obj.select("div.list_basis>div>div")
            in_list = list(range(0, len(box_list)))
            print(f"{self.search_site} BOX_LIST : {len(box_list)}")
            # 광고 물품 제외, 다른 오픈마켓의 상품을 중게하는 물품 제외
            for idx, box in enumerate(box_list):
                if ("광고" in box.find("div", {"class": "basicList_price_area__K7DDT"}).text) or ("판매처" in box.find("div", {"class": "basicList_price_area__K7DDT"}).text):
                    continue
                href = box.find("a", {"class":"thumbnail_thumb__Bxb6Z"})["href"]
                
                in_url_list.append(href)
        elif self.search_site == "ssg":
            box_list = bs_obj.select('#idProductImg>li')

            for box in box_list:
                box_a = box.select('div.thmb>a')[0]
                try:
                    product_url = box_a["href"]
                    in_url_list.append(product_url)
                except:
                    pass
        
        return in_url_list

    # 상품 페이지로 이동, 데이터 크롤링
    def data_crawler(self, url):
        if self.search_site == "gmarket":
            self.driver.get(url)
        elif self.search_site == "naver":
            self.driver.get(url)
        elif self.search_site == "ssg":
            self.driver.get(f"https://www.ssg.com/{url}")

        self.driver.implicitly_wait(7)

        time.sleep(7)
        
        # 상품에 대한 데이터는 바로 데이터프레임화 하여 저장
        # 리뷰에 관한 데이터는 리스트로 저장 후 데이터프레임화
        rTitle_list = list()
        rText_list = list()
        rGrade_list = list()
        rOption_list = list()

        rDate_list = list()
        rCdate_list = list()
        rSource_list = list()
        
        # 지마켓 수집
        if self.search_site == "gmarket":
            # 상품 정보
            goods_box = self.driver.find_element(by=By.ID, value='itemcase_basic')
            
            # 상품 이름
            goods_title = goods_box.find_element(by=By.CLASS_NAME, value='itemtit').text
            
            # 상품 이미지
            thum_box = self.driver.find_element(by=By.CLASS_NAME, value='thumb-gallery')
            goods_img = thum_box.find_element(by=By.TAG_NAME, value='img').get_attribute('src')

            # 실제 가격
            temp_sprice = goods_box.find_element(by=By.CLASS_NAME, value='price_real').text
            try:
                goods_sprice = float(temp_sprice.replace(",", "").replace("원", ""))
            except:
                goods_sprice = 0

            # 상품 URL
            goods_url = self.driver.current_url

            try:
                # 원래 가격
                temp_oprice = goods_box.find_element(by=By.CLASS_NAME, value='price_original').text
                goods_oprice = temp_oprice.replace(",", "").replace("원", "")
            except:
                goods_oprice = goods_sprice
                
            # 상품 카테고리
            temp_category = self.driver.find_element(by=By.CLASS_NAME, value='location-navi').text
            goods_category = temp_category.replace("\n열기", "")
            
            # 배송비
            temp_fee = self.driver.find_element(by=By.CLASS_NAME, value='list-item-delivery').text
            if "Smile 배송" in temp_fee:
                try:
                    goods_fee = temp_fee.split("설정하기\n")[1].split("\n무료체험")[0]
                except:
                    goods_fee = temp_fee
            else:
                try:
                    goods_fee = temp_fee.split("\n열기")[0].split("안내글 토글\n\n")[1]
                except:
                    goods_fee = temp_fee
    
            # 적립금
            temp_cashback = self.driver.find_element(by=By.CLASS_NAME, value='list-item-smileclub').text
            temp_cashback_percent = temp_cashback.split("%")[0][-1]
            goods_cashback = float(goods_sprice)*float(temp_cashback_percent)/100
            
            # 리뷰 수
            try:
                temp_review = self.driver.find_element(by=By.XPATH, value=f'//*[@id="txtReviewTotalCount"]').text
                goods_reviewCount = float(temp_review.replace(",", ""))
            except:
                goods_reviewCount = 0

            mobile_url = goods_url.replace("//item", "//mitem")
            
            self.driver.get(mobile_url)
            
            self.driver.implicitly_wait(10)
            
            time.sleep(5)
            
            # 상품평 목록 출력
            self.driver.find_element(by=By.XPATH, value=f'//*[@id="mainTab1"]/button').click()

            time.sleep(5)
            try:
                temp_reviewGrade = self.driver.find_element(by=By.CLASS_NAME, value='text__tatal-review').text
            except NoSuchElementException:
                self.driver.find_element(by=By.XPATH, value=f'//*[@id="mainTab1"]/button').click()

                time.sleep(7)

                temp_reviewGrade = self.driver.find_element(by=By.CLASS_NAME, value='text__tatal-review').text                
            goods_reviewGrade = float(temp_reviewGrade.split("평점")[-1])
            
            print(f"GMARKET : {goods_title}")
            gTemp_df = pd.DataFrame({
                'TITLE':[goods_title],
                'IMAGE':[goods_img],
                'URL':[goods_url],
                'SALEPRICE':[goods_sprice],
                'ORIPRICE':[goods_oprice],
                'CATEGORY':[goods_category],
                'DELIVERYFEE':[goods_fee],
                'CASHBACK':[goods_cashback],
                'INFO':[''],
                'GRADE':[goods_reviewGrade],
                'REVIEWCOUNT':[goods_reviewCount],
                'SOURCE':['gmarket']
            })
            
            self.goods_df = pd.concat([self.goods_df, gTemp_df], ignore_index=True)
            
            try:
                review_premiumCount = self.driver.find_element(by=By.XPATH, 
                                                               value=f'//*[@id="photoReviewTab"]/a/span').text
                review_premiumCount = int(review_premiumCount.replace(",", ""))
            except:
                review_premiumCount = 0

            if review_premiumCount > 50:
                review_premiumCount = 50
            
            if review_premiumCount != 0:
                try:
                    self.driver.find_element(by=By.XPATH, value=f'//*[@id="coreReviewArea"]/div[1]/div[3]/ul/li[2]').click()
                except NoSuchElementException:
                    self.driver.find_element(by=By.XPATH, value=f'//*[@id="mainTab1"]/button').click()

                    time.sleep(2)

                    self.driver.find_element(by=By.XPATH, value=f'//*[@id="coreReviewArea"]/div[1]/div[3]/ul/li[2]').click()

                for i in range(1, review_premiumCount+1):
                    try:
                        review_box = self.driver.find_element(by=By.XPATH,
                                                              value=f'//*[@id="photoReviewArea"]/ul/li[{i}]').text
                    except NoSuchElementException:
                        try:
                            time.sleep(3)
                            review_box = self.driver.find_element(by=By.XPATH,
                                                                  value=f'//*[@id="photoReviewArea"]/ul/li[{i}]').text
                        except NoSuchElementException:
                            time.sleep(3)
                            review_box = self.driver.find_element(by=By.XPATH,
                                                                  value=f'//*[@id="photoReviewArea"]/ul/li[{i}]').text
                    if len(review_box.split("\n")) == 7:
                        review_date = review_box.split("\n")[1][:10]

                        temp_reviewGrade = review_box.split("\n")[0]

                        review_option = review_box.split("\n")[3]
                    elif len(review_box.split("\n")) == 8:
                        review_date = review_box.split("\n")[2][:10]

                        temp_reviewGrade = review_box.split("\n")[1]

                        review_option = review_box.split("\n")[4]
                    elif len(review_box.split("\n")) == 6:
                        review_date = review_box.split("\n")[1][:10]

                        temp_reviewGrade = review_box.split("\n")[0]

                        review_option = review_box.split("\n")[3]                        
                    review_grade = temp_reviewGrade[-2]

                    temp_text = review_box.split("\n")[-2]
                    review_text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", temp_text)
                    
                    if i % 10 == 0 and i != review_premiumCount:
                        scroll_down(self.driver)

                        self.driver.find_element(by=By.XPATH, value=f'//*[@id="photoReviewMoreBtn"]/a').click()

                        time.sleep(2)
                    
                    if review_text != "":
                        rTitle_list.append(goods_title)
                        rText_list.append(review_text)
                        
                        rOption_list.append(review_option)
                        rGrade_list.append(review_grade)
                        rDate_list.append(review_date)
                        rCdate_list.append(self.nowdate3)
                    
            rTemp_df = pd.DataFrame({
                'title':rTitle_list,
                'content':rText_list,
                'option':rOption_list,
                'grade':rGrade_list,
                'date':rDate_list,
                'collect_date':rCdate_list
            })    
            rTemp_df['source'] = 'gmarket'
            self.review_df = pd.concat([self.review_df, rTemp_df], ignore_index=True)
        # 네이버 수집
        elif self.search_site == "naver":
            scroll_down(self.driver)
            time.sleep(2)

            # 상품명
            goods_title = self.driver.find_element(by=By.XPATH, value='//*[@id="content"]/div/div[2]/div[2]/fieldset/' +\
                                                   'div[1]/div[1]').text            
            # 상품 이미지
            goods_img = self.driver.find_element(by=By.XPATH, 
                                                 value='//*[@id="content"]/div/div[2]/div[1]/div[1]/div[1]/img').get_attribute('src')
            # 상품 URL
            goods_url = self.driver.current_url
            
            # 상품 가격(할인 가격, 정상 가격)
            temp_sprice = self.driver.find_element(by=By.XPATH, value='//*[@id="content"]/div/div[2]/div[2]/fieldset/' +\
                                                   'div[1]/div[2]/div/strong/span[2]').text
            goods_sprice = float(temp_sprice.replace(",", ""))
            try:
                temp_oprice = self.driver.find_element(by=By.XPATH, value=f'//*[@id="content"]/div/div[2]/div[2]/fieldset' + \
                                                  '/div[1]/div[2]/div/del/span[2]')
                goods_oprice = float(temp_oprice.replace(",", ""))
            except:
                goods_oprice = goods_sprice

            # 상품 카테고리
            temp_category = self.driver.find_element(by=By.CLASS_NAME, value='_1J9J3q04Tn').text
            try:
                goods_category = temp_category.split("\n전체")[0]
            except:
                goods_category = temp_category
                
            # 상품 배송비
            try:
                temp_fee = self.driver.find_element(by=By.CLASS_NAME, value='bd_3uare').text
                goods_fee = temp_fee.replace(",", "")
            except NoSuchElementException:
                goods_fee = 0

            # 상품 적립금
            temp_cashback = self.driver.find_element(by=By.CLASS_NAME, value='_1vMhLvKfMe').text
            try:
                goods_cashback = float(temp_cashback.split("최대")[1][:-1].strip())
            except:
                goods_cashback = 0
            
            # 상품 정보
            try:
                goods_info1 = self.driver.find_element(by=By.XPATH, 
                                                       value='//*[@id="INTRODUCE"]/div/div[3]/div/div[1]/table').text
                goods_info2 = self.driver.find_element(by=By.XPATH, 
                                                       value='//*[@id="INTRODUCE"]/div/div[3]/div/div[2]/div/table').text
                goods_info = f"{goods_info1}\n{goods_info2}"
            except:
                try:
                    goods_info1 = self.driver.find_element(by=By.XPATH, 
                                                           value='//*[@id="INTRODUCE"]/div/div[4]/div/div[1]/table').text
                    goods_info2 = self.driver.find_element(by=By.XPATH, 
                                                           value='//*[@id="INTRODUCE"]/div/div[4]/div/div[2]/div/table').text
                    goods_info = f"{goods_info1}\n{goods_info2}"
                except:
                    goods_info = "None"     

            # 리뷰 총점, 리뷰 총 개수
            try:
                temp_grade = self.driver.find_element(by=By.XPATH, value='//*[@id="REVIEW"]/' +\
                                                      'div/div[2]/div[1]/div/div[1]/div/span').text
                temp_grade = temp_grade.split("중")[1]
                goods_reviewGrade = float(temp_grade.split("점")[0].strip())

                temp_review = self.driver.find_element(by=By.XPATH, value='//*[@id="REVIEW"]/' +\
                                                       'div/div[2]/div[1]/div/div[2]/div/span[2]').text
                goods_reviewCount = int(temp_review.split("개")[0].replace(",", ""))
            except:
                goods_reviewGrade = 0
                goods_reviewCount = 0

            print(f"NAVER : {goods_title}")
            gTemp_df = pd.DataFrame({
                'TITLE':[goods_title],
                'IMAGE':[goods_img],
                'URL':[goods_url],
                'SALEPRICE':[goods_sprice],
                'ORIPRICE':[goods_oprice],
                'CATEGORY':[goods_category],
                'DELIVERYFEE':[goods_fee],
                'CASHBACK':[goods_cashback],
                'INFO':[goods_info],
                'GRADE':[goods_reviewGrade],
                'REVIEWCOUNT':[goods_reviewCount],
                'SOURCE':['naver']
            })
            
            self.goods_df = pd.concat([self.goods_df, gTemp_df], ignore_index=True)
            
            # 페이지 정보 >> 리뷰 수집 개수를 제한하는 부분
            review_page = math.ceil(goods_reviewCount/20)
            if review_page > 50:
                review_page = 50

            review_remain = int(goods_reviewCount%20)
            if review_remain == 0:
                review_remain = 20    

            self.driver.find_element(by=By.XPATH, value='//*[@id="REVIEW"]/div/div[3]/div[1]/div[1]/ul/li[2]/a').send_keys(Keys.RETURN)

            time.sleep(3)

            # 리뷰 수집
            for i in range(1, review_page+1):
                current_page = i

                if current_page == 1:
                    pass
                elif current_page < 6:
                    self.driver.find_element(by=By.XPATH,
                                             value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/div/div/a[{current_page + 6}]').click()
                    time.sleep(4)
                else:
                    self.driver.find_element(by=By.XPATH,
                                             value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/div/div/a[12]').click()
                    time.sleep(4)

                if current_page == review_page:
                    for j in range(1, review_remain+1):
                        try:
                            review_box = self.driver.find_element(by=By.XPATH, 
                                                                  value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                                  '/div/div/div/div[1]/div/div[1]/div[2]/div/span')
                            if review_box.text in ['재구매', '한달사용기', 'BEST']:
                                review_box = self.driver.find_element(by=By.XPATH, 
                                                                      value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                                      '/div/div/div/div[1]/div/div[1]/div[2]/div/span[2]')
                        except NoSuchElementException:
                            try:
                                review_box = self.driver.find_element(by=By.XPATH, 
                                                                    value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                                    '/div/div/div/div[1]/div/div[1]/div[2]/div/span[2]')
                            except NoSuchElementException:
                                review_box = self.driver.find_element(by=By.XPATH, 
                                                                    value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                                    '/div/div/div/div[1]/div/div/div[2]/div/span')

                        # 리뷰 텍스트 전처리(특수문자 제거)
                        temp_text = review_box.text.replace("\n", " ")
                        review_text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", temp_text)
                        try:
                            grade = self.driver.find_element(by=By.XPATH, 
                                                            value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                            '/div/div/div/div[1]/div/div[1]/div[1]/div[2]/div[1]')
                        except NoSuchElementException:
                            grade = self.driver.find_element(by=By.XPATH, 
                                                            value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                            '/div/div/div/div[1]/div/div/div[1]/div[2]/div[1]')

                        grade_temp = grade.text
                        review_grade = float(grade_temp.replace("평점\n", ""))
                        try:
                            date_temp = self.driver.find_element(by=By.XPATH, 
                                                            value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                            '/div/div/div/div[1]/div/div[1]/div[1]/div[2]/div[2]/span').text
                        except NoSuchElementException:
                            date_temp = self.driver.find_element(by=By.XPATH, 
                                                            value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                            '/div/div/div/div[1]/div/div/div[1]/div[2]/div[2]/span').text

                        try:
                            review_option = self.driver.find_element(by=By.XPATH,
                                                                    value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' +\
                                                                     '/div/div/div/div[1]/div/div[1]/div[1]/div[2]/div[3]').text
                        except NoSuchElementException:
                            try:
                                review_option = self.driver.find_element(by=By.XPATH,
                                                                    value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' +\
                                                                     '/div/div/div/div[1]/div/div/div[1]/div[2]/div[3]').text
                            except:
                                review_option = "옵션없음"
                        
                        rText_list.append(review_text)
                        rGrade_list.append(review_grade)

                        rOption_list.append(review_option)
                        rTitle_list.append(goods_title)
                        rDate_list.append(f'20{date_temp[:-1]}')
                        rCdate_list.append(self.nowdate3)
                else:
                    for j in range(1, 21):
                        try:
                            review_box = self.driver.find_element(by=By.XPATH, 
                                                                  value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                                  '/div/div/div/div[1]/div/div[1]/div[2]/div/span')
                            if review_box.text in ['재구매', '한달사용기', 'BEST']:
                                review_box = self.driver.find_element(by=By.XPATH, 
                                                                      value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                                      '/div/div/div/div[1]/div/div[1]/div[2]/div/span[2]')
                        except NoSuchElementException:
                            try:
                                review_box = self.driver.find_element(by=By.XPATH, 
                                                                    value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                                    '/div/div/div/div[1]/div/div[1]/div[2]/div/span[2]')
                            except NoSuchElementException:
                                review_box = self.driver.find_element(by=By.XPATH, 
                                                                    value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                                    '/div/div/div/div[1]/div/div/div[2]/div/span')

                        temp_text = review_box.text.replace("\n", " ")
                        review_text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", temp_text)
                        try:
                            grade = self.driver.find_element(by=By.XPATH, 
                                                            value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                            '/div/div/div/div[1]/div/div[1]/div[1]/div[2]/div[1]')
                        except NoSuchElementException:
                            grade = self.driver.find_element(by=By.XPATH, 
                                                            value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                            '/div/div/div/div[1]/div/div/div[1]/div[2]/div[1]')

                        grade_temp = grade.text
                        review_grade = float(grade_temp.replace("평점\n", ""))
                        try:
                            date_temp = self.driver.find_element(by=By.XPATH, 
                                                            value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                            '/div/div/div/div[1]/div/div[1]/div[1]/div[2]/div[2]/span').text
                        except NoSuchElementException:
                            date_temp = self.driver.find_element(by=By.XPATH, 
                                                            value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' + \
                                                            '/div/div/div/div[1]/div/div/div[1]/div[2]/div[2]/span').text

                        try:
                            review_option = self.driver.find_element(by=By.XPATH,
                                                                    value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' +\
                                                                     '/div/div/div/div[1]/div/div[1]/div[1]/div[2]/div[3]').text
                        except NoSuchElementException:
                            try:
                                review_option = self.driver.find_element(by=By.XPATH,
                                                                    value=f'//*[@id="REVIEW"]/div/div[3]/div[2]/ul/li[{j}]' +\
                                                                     '/div/div/div/div[1]/div/div/div[1]/div[2]/div[3]').text
                            except:
                                review_option = "옵션없음"
                            
                        rText_list.append(review_text)
                        rGrade_list.append(review_grade)

                        rOption_list.append(review_option)
                        rTitle_list.append(goods_title)
                        rDate_list.append(f'20{date_temp[:-1]}')
                        rCdate_list.append(self.nowdate3)
                        
                                        
            rTemp_df = pd.DataFrame({
                'TITLE':rTitle_list,
                'CONTENT':rText_list,
                'OPTION':rOption_list,
                'GRADE':rGrade_list,
                'DATE':rDate_list,
                'COLLECT_DATE':rCdate_list
            })       
            rTemp_df['SOURCE'] = 'naver'
            self.review_df = pd.concat([self.review_df, rTemp_df], ignore_index=True)
        elif self.search_site == "ssg":
            try:
                goods_url = self.driver.current_url

                goods_opt = self.driver.find_element(by=By.XPATH, value='//*[@id="dealItemInfoTab"]/div/div[1]')
                # goods_opt = self.driver.find_element(by=By.CLASS_NAME, value='cdtl_info_sel')

                # 상품 카테고리
                temp_category = self.driver.find_element(by=By.ID, value='location').text
                goods_category = temp_category.replace("\n", "")

                ul = goods_opt.find_element(by=By.CLASS_NAME, value='chd_select_lst')
                
                for k in range(1, 100):
                    self.driver.find_element(by=By.XPATH, value='//*[@id="_cdtl_dtlcont_wrap"]/div[1]/div/div[1]/ul/li[2]/a').click()
                    
                    goods_opt.click()

                    try:
                        ul.find_element(by=By.XPATH, value=f'li[{k}]').click()
                    except ElementClickInterceptedException:
                        continue
                    except NoSuchElementException:
                        break

                    time.sleep(2)

                    # 상품명
                    goods_title = goods_opt.find_element(by=By.CLASS_NAME, value='txt').text

                    # 상품 이미지
                    goods_img = goods_opt.find_element(by=By.XPATH, value='//*[@class="thmb"]/img').get_attribute('src')

                    goods_box = self.driver.find_element(by=By.ID, value='info_contents')

                    # 실제 가격
                    temp_sprice = goods_box.find_element(by=By.CLASS_NAME, value='ssg_price').text
                    try:
                        goods_sprice = float(temp_sprice.replace(",", "").replace("원", ""))
                    except:
                        goods_sprice = 0
                        
                    goods_oprice = goods_sprice

                    # 상품 적립금
                    temp_cashback = goods_box.find_element(by=By.CLASS_NAME, value='cdtl_mmbr_txt').text

                    try:
                        goods_cashback = float(temp_cashback.split(" ")[1][:-1])
                    except:
                        goods_cashback = 0

                    # 배송비
                    try:
                        fee_box = goods_box.find_element(by=By.CLASS_NAME, value='cdtl_delivery_fee')
                        temp_fee = fee_box.find_element(by=By.CLASS_NAME, varslue='ssg_price'),text
                        goods_fee = float(temp_fee.replace(","))
                    except NoSuchElementException:
                        goods_fee = 0
                        
                    # 리뷰 수
                    try:
                        temp_review = self.driver.find_element(by=By.CLASS_NAME, value="t_review").text
                        
                        goods_reviewCount = float(temp_review.split(" ")[1][:-1].replace(",", ""))
                    except:
                        goods_reviewCount = 0
                        
                    temp_reviewGrade = self.driver.find_element(by=By.CLASS_NAME, value='cdtl_star_score').text
                    goods_reviewGrade = float(temp_reviewGrade)

                    goods_info = self.driver.find_element(by=By.XPATH, value='//*[@id="info_contents"]/div[2]/div[2]').text

                    print(f"SSG : {goods_title}")
                    gTemp_df = pd.DataFrame({
                        'TITLE':[goods_title],
                        'IMAGE':[goods_img],
                        'URL':[goods_url],
                        'SALEPRICE':[goods_sprice],
                        'ORIPRICE':[goods_oprice],
                        'CATEGORY':[goods_category],
                        'DELIVERYFEE':[goods_fee],
                        'CASHBACK':[goods_cashback],
                        'INFO':[goods_info],
                        'GRADE':[goods_reviewGrade],
                        'REVIEWCOUNT':[goods_reviewCount],
                        'SOURCE':['ssg']
                    })

                    self.goods_df = pd.concat([self.goods_df, gTemp_df], ignore_index=True)

                    time.sleep(2)

                    self.driver.find_element(by=By.XPATH, value=f'//*[@id="_cdtl_dtlcont_wrap"]/div[1]/div/div[1]/ul/li[2]/a').click()
                    
                    self.driver.find_element(by=By.XPATH, value=f'//*[@id="cmt_select_sort"]').click()
                    time.sleep(2)
                    self.driver.find_element(by=By.XPATH, value=f'//*[@id="cmt_select_sort"]/div/div/ul/li[2]').click()

                    time.sleep(1)

                    review_page = math.ceil(goods_reviewCount/10)
                    if review_page > 5:
                        review_page = 5
                        review_remain = 10
                    else:
                        review_remain = int(goods_reviewCount%10)
                        if review_remain == 0:
                            review_remain = 10     

                    first_box = self.driver.find_element(by=By.XPATH, 
                                                    value=f'//*[@id="cdtl_cmt_tbody"]/tr[1]/td[1]/div')

                    try:
                        first_option = first_box.find_element(by=By.CLASS_NAME, value='cdtl_cmt_tx2').text
                    except NoSuchElementException:
                        first_option = "없음"

                    for i in range(1, review_page+1):
                        current_page = i
                        temp_page = current_page % 10
                        if temp_page == 0:
                            temp_page = 10
                        elif temp_page == 1:
                            temp_page = 11        

                        if current_page == 1:
                            pass
                        elif current_page == 2:
                            self.driver.find_element(by=By.XPATH,
                                                    value=f'//*[@id="comment_navi_area"]/a[1]').click()
                            time.sleep(4)        
                        elif current_page < 12:
                            self.driver.find_element(by=By.XPATH,
                                                    value=f'//*[@id="comment_navi_area"]/a[{current_page}]').click()
                            time.sleep(4)
                        else:
                            self.driver.find_element(by=By.XPATH,
                                                    value=f'//*[@id="comment_navi_area"]/a[{temp_page + 1}]').click()
                            time.sleep(4)

                        if current_page == review_page:
                            for j in range(1, review_remain+1):
                                review_box = self.driver.find_element(by=By.XPATH, 
                                                                    value=f'//*[@id="cdtl_cmt_tbody"]/tr[{2*j - 1}]/td[1]/div')
                                
                                temp_text = review_box.find_element(by=By.CLASS_NAME, value='cdtl_cmt_tx').text
                                review_text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", temp_text)

                                temp_grade = review_box.find_element(by=By.CLASS_NAME, value='ico_star').text
                                review_grade = float(temp_grade)
                                
                                review_date = review_box.find_element(by=By.CLASS_NAME, value='user_date').text
                                
                                if first_option == "없음":
                                    review_option = "옵션없음"
                                else:
                                    try:
                                        review_option = review_box.find_element(by=By.CLASS_NAME, value='cdtl_cmt_tx2').text
                                    except NoSuchElementException:
                                        review_option = "옵션없음"
                                

                                rText_list.append(review_text)
                                rGrade_list.append(review_grade)

                                rOption_list.append(review_option)
                                rTitle_list.append(goods_title)
                                rDate_list.append(review_date)
                                rCdate_list.append(self.nowdate3)
                        else:
                            for j in range(1, 11):
                                review_box = self.driver.find_element(by=By.XPATH, 
                                                                value=f'//*[@id="cdtl_cmt_tbody"]/tr[{2*j - 1}]/td[1]/div')
                                
                                temp_text = review_box.find_element(by=By.CLASS_NAME, value='cdtl_cmt_tx').text
                                review_text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", temp_text)

                                temp_grade = review_box.find_element(by=By.CLASS_NAME, value='ico_star').text
                                review_grade = float(temp_grade)
                                
                                review_date = review_box.find_element(by=By.CLASS_NAME, value='user_date').text
                                
                                if first_option == "없음":
                                    review_option = "옵션없음"
                                else:
                                    try:
                                        review_option = review_box.find_element(by=By.CLASS_NAME, value='cdtl_cmt_tx2').text
                                    except NoSuchElementException:
                                        review_option = "옵션없음"

                                rText_list.append(review_text)
                                rGrade_list.append(review_grade)

                                rOption_list.append(review_option)
                                rTitle_list.append(goods_title)
                                rDate_list.append(review_date)
                                rCdate_list.append(self.nowdate3)
                                
                    rTemp_df = pd.DataFrame({
                        'title':rTitle_list,
                        'content':rText_list,
                        'option':rOption_list,
                        'grade':rGrade_list,
                        'date':rDate_list,
                        'collect_date':rCdate_list
                    })    
                    rTemp_df['SOURCE'] = 'ssg'
                    self.review_df = pd.concat([self.review_df, rTemp_df], ignore_index=True)
            except NoSuchElementException:
                goods_box = self.driver.find_element(by=By.CLASS_NAME, value='cdtl_col_rgt')

                # 상품 이름
                goods_title = goods_box.find_element(by=By.CLASS_NAME, value='cdtl_info_tit_name').text

                # 상품 이미지
                thum_box = self.driver.find_element(by=By.CLASS_NAME, value='cdtl_item_image')
                goods_img = thum_box.find_element(by=By.TAG_NAME, value='img').get_attribute('src')

                # 실제 가격
                temp_sprice = goods_box.find_element(by=By.CLASS_NAME, value='ssg_price').text
                try:
                    goods_sprice = float(temp_sprice.replace(",", "").replace("원", ""))
                except:
                    goods_sprice = 0

                goods_url = self.driver.current_url

                try:
                    # 원래 가격
                    temp_oprice = goods_box.find_element(by=By.CLASS_NAME, value='cdtl_old_price').text
                    goods_oprice = temp_oprice.replace(",", "").replace("원", "")
                except:
                    goods_oprice = goods_sprice

                # 상품 카테고리
                temp_category = self.driver.find_element(by=By.ID, value='location').text
                goods_category = temp_category.replace("\n", "")

                # 배송비
                try:
                    fee_box = goods_box.find_element(by=By.CLASS_NAME, value='cdtl_delivery_fee')
                    temp_fee = fee_box.find_element(by=By.CLASS_NAME, varslue='ssg_price'),text
                    goods_fee = float(temp_fee.replace(","))
                except NoSuchElementException:
                    goods_fee = 0

                # 상품 적립금
                temp_cashback = self.driver.find_element(by=By.CLASS_NAME, value='cdtl_mmbr_txt').text

                try:
                    goods_cashback = float(temp_cashback.split(" ")[1][:-1])
                except:
                    goods_cashback = 0

                # 리뷰 수
                try:
                    temp_review = self.driver.find_element(by=By.ID, value="postngNlistCnt").text
                    goods_reviewCount = float(temp_review.replace(",", ""))
                except:
                    goods_reviewCount = 0

                # 리뷰 점수
                try:
                    temp_reviewGrade = self.driver.find_element(by=By.CLASS_NAME, value='cdtl_star_score').text
                    goods_reviewGrade = float(temp_reviewGrade)
                except:
                    goods_reviewGradew = 0
                try:
                    goods_info = self.driver.find_element(by=By.XPATH, value='//*[@id="item_detail"]/div[1]/div[4]/div[2]/div').text
                except NoSuchElementException:
                    goods_info = ''

                print(f"SSG : {goods_title}")
                gTemp_df = pd.DataFrame({
                    'TITLE':[goods_title],
                    'IMAGE':[goods_img],
                    'URL':[goods_url],
                    'SALEPRICE':[goods_sprice],
                    'ORIPRICE':[goods_oprice],
                    'CATEGORY':[goods_category],
                    'DELIVERYFEE':[goods_fee],
                    'CASHBACK':[goods_cashback],
                    'INFO':[goods_info],
                    'GRADE':[goods_reviewGrade],
                    'REVIEWCOUNT':[goods_reviewCount],
                    'SOURCE':['ssg']
                })

                self.goods_df = pd.concat([self.goods_df, gTemp_df], ignore_index=True)
            
                time.sleep(2)

                self.driver.find_element(by=By.XPATH, value=f'//*[@id="_cdtl_dtlcont_wrap"]/div[1]/div/div[1]/ul/li[2]/a').click()
                
                self.driver.find_element(by=By.XPATH, value=f'//*[@id="cmt_select_sort"]').click()
                time.sleep(2)
                self.driver.find_element(by=By.XPATH, value=f'//*[@id="cmt_select_sort"]/div/div/ul/li[2]').click()

                time.sleep(1)

                review_page = math.ceil(goods_reviewCount/10)
                if review_page > 5:
                    review_page = 5
                    review_remain = 10
                else:
                    review_remain = int(goods_reviewCount%10)
                    if review_remain == 0:
                        review_remain = 10     

                first_box = self.driver.find_element(by=By.XPATH, 
                                                value=f'//*[@id="cdtl_cmt_tbody"]/tr[1]/td[1]/div')

                try:
                    first_option = first_box.find_element(by=By.CLASS_NAME, value='cdtl_cmt_tx2').text
                except NoSuchElementException:
                    first_option = "없음"

                for i in range(1, review_page+1):
                    current_page = i
                    temp_page = current_page % 10
                    if temp_page == 0:
                        temp_page = 10
                    elif temp_page == 1:
                        temp_page = 11        

                    if current_page == 1:
                        pass
                    elif current_page == 2:
                        self.driver.find_element(by=By.XPATH,
                                                value=f'//*[@id="comment_navi_area"]/a[1]').click()
                        time.sleep(4)        
                    elif current_page < 12:
                        self.driver.find_element(by=By.XPATH,
                                                value=f'//*[@id="comment_navi_area"]/a[{current_page}]').click()
                        time.sleep(4)
                    else:
                        self.driver.find_element(by=By.XPATH,
                                                value=f'//*[@id="comment_navi_area"]/a[{temp_page + 1}]').click()
                        time.sleep(4)

                    if current_page == review_page:
                        for j in range(1, review_remain+1):
                            review_box = self.driver.find_element(by=By.XPATH, 
                                                                value=f'//*[@id="cdtl_cmt_tbody"]/tr[{2*j - 1}]/td[1]/div')
                            
                            temp_text = review_box.find_element(by=By.CLASS_NAME, value='cdtl_cmt_tx').text
                            review_text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", temp_text)

                            temp_grade = review_box.find_element(by=By.CLASS_NAME, value='ico_star').text
                            review_grade = float(temp_grade)
                            
                            review_date = review_box.find_element(by=By.CLASS_NAME, value='user_date').text
                            
                            if first_option == "없음":
                                review_option = "옵션없음"
                            else:
                                try:
                                    review_option = review_box.find_element(by=By.CLASS_NAME, value='cdtl_cmt_tx2').text
                                except NoSuchElementException:
                                    review_option = "옵션없음"
                            

                            rText_list.append(review_text)
                            rGrade_list.append(review_grade)

                            rOption_list.append(review_option)
                            rTitle_list.append(goods_title)
                            rDate_list.append(review_date)
                            rCdate_list.append(self.nowdate3)
                    else:
                        for j in range(1, 11):
                            review_box = self.driver.find_element(by=By.XPATH, 
                                                            value=f'//*[@id="cdtl_cmt_tbody"]/tr[{2*j - 1}]/td[1]/div')
                            
                            temp_text = review_box.find_element(by=By.CLASS_NAME, value='cdtl_cmt_tx').text
                            review_text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", temp_text)

                            temp_grade = review_box.find_element(by=By.CLASS_NAME, value='ico_star').text
                            review_grade = float(temp_grade)
                            
                            review_date = review_box.find_element(by=By.CLASS_NAME, value='user_date').text
                            
                            if first_option == "없음":
                                review_option = "옵션없음"
                            else:
                                try:
                                    review_option = review_box.find_element(by=By.CLASS_NAME, value='cdtl_cmt_tx2').text
                                except NoSuchElementException:
                                    review_option = "옵션없음"

                            rText_list.append(review_text)
                            rGrade_list.append(review_grade)

                            rOption_list.append(review_option)
                            rTitle_list.append(goods_title)
                            rDate_list.append(review_date)
                            rCdate_list.append(self.nowdate3)
                            
                rTemp_df = pd.DataFrame({
                    'title':rTitle_list,
                    'content':rText_list,
                    'option':rOption_list,
                    'grade':rGrade_list,
                    'date':rDate_list,
                    'collect_date':rCdate_list
                })    
                rTemp_df['SOURCE'] = 'ssg'
                self.review_df = pd.concat([self.review_df, rTemp_df], ignore_index=True)

        time.sleep(2)

        # 데이터 1차 저장
    def data_save(self, save_path, product):
        goods_df = self.goods_df
        review_df = self.review_df
        
        goods_df.to_csv(path_or_buf=f'{save_path}/GOODS_RAW_{self.search_site}_{product}_{self.nowdate}.csv',
                        index=False, encoding='utf-8')
        review_df.to_csv(path_or_buf=f'{save_path}/REVIEW_RAW_{self.search_site}_{product}_{self.nowdate}.csv',
                         index=False, encoding='utf-8')

        print(f"{self.search_site} DATA_SAVE : {product} / PRODUCT : {len(goods_df)}")

        self.driver.quit()