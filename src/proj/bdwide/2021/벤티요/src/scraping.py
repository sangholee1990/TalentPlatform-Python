import requests
import os 
import csv
import pandas as pd
import datetime
from tqdm import tqdm
 

class WeatherApi:
    """ 미국 50개 주 월별 평균 기온 요청 클래스"""
    def __init__(self):
        self.baseurl = 'https://www.ncdc.noaa.gov/cag/statewide/time-series/'
        self.begyear = 2010 # 시작 년도
        self.endyear = 2020 # 끝 년도 
        self.begmonth = 1 # 시작 월
        self.states = self.read_states()


    def read_states(self):
        states_dict = {}
        with open('states.txt', 'r') as f:
            states = f.readlines()
            for state in states:
                values = state.split(',')
                if len(values) == 2:
                    states_dict[values[0]] = values[1].strip()
        return states_dict

    def download_csv(self, begyear=None, endyear=None):
        if begyear != None:
            self.begyear = begyear
        if endyear != None:
            self.endyear = endyear 


        # USweather 폴더 생성 
        os.makedirs('USweather', exist_ok=True)
        os.chdir('USweather')

        # 현재시간으로 폴더생성 - 데이터 중복 방지 
        current_time = datetime.datetime.now()
        current_time = str(current_time).split(',')[0].replace(':','-').strip()
        os.makedirs(current_time)
        os.chdir(current_time)

        for num, sname in tqdm(self.states.items()):
            with requests.Session() as s:
                url = self.baseurl+num+'-tavg-all-1-'+str(self.begyear)+'-'+str(self.endyear)+'.csv?base_prd=true&begbaseyear='+str(self.begyear)+'&endbaseyear='+str(self.endyear)
                download = s.get(url)
                decoded_content = download.content.decode('utf-8')
                cr = csv.reader(decoded_content.splitlines(), delimiter=',')
                my_list = list(cr)
                df = pd.DataFrame(my_list[4:])
                df.to_csv(sname+'.csv')



    

if __name__ == "__main__": 
    w = WeatherApi()
    w.download_csv(begyear=2008, endyear=2021)


        

    
    

        