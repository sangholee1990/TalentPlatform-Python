
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List
import pwlf
import pwlf as pwlf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# import read_data
import glob

# ============================================
# 보조
# ============================================
def extract_city(city):

    global india_data
    india_data=india_data[(india_data["sector"]=="Residential")]
    city_data=india_data[["city","value (KtCO2 per day)"]]
    city_data=city_data.rename(columns={"value (KtCO2 per day)":"Total"})
    city_data["Dummy"]=np.nan
    return city_data[city_data["city"]==city]

def extract_city_ERA5(city):
    ERA5 = _ERA5[(_ERA5["NAME_1"]==city)]
    return ERA5

#####################################
############## utils ################
#####################################

def fit_piecewise_func(x: np.array, y: np.array):
    '''
    modified from Rohith's script
    '''
    pwlf_dic = {}
    n = len(x)

    # 1seg
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    pred = model.predict(x.reshape(-1, 1))

    pwlf_dic['bp0-r2'] = r2_score(y, pred)
    pwlf_dic['bp0-rmse'] = np.sqrt(mean_squared_error(y, pred))
    pwlf_dic['bp0-aic'] = (n * np.log(mean_squared_error(y, pred))) + (2 * 2)
    pwlf_dic['bp0-bic'] = n * np.log(mean_squared_error(y, pred)) + (2 * np.log(n))
    pwlf_dic['bp0-slope'] = model.coef_

    # 2seg
    my_pwlf2 = pwlf.PiecewiseLinFit(x, y)
    res2 = my_pwlf2.fit(2, seed=42)
    pred_2seg = my_pwlf2.predict(x)

    k = my_pwlf2.n_parameters
    pwlf_dic['bp1-r2'] = my_pwlf2.r_squared()
    pwlf_dic['bp1-rmse'] = np.sqrt(my_pwlf2.ssr / n)
    pwlf_dic['bp1-aic'] = n * np.log(my_pwlf2.ssr / n) + (k * 2)
    pwlf_dic['bp1-bic'] = n * np.log(my_pwlf2.ssr / n) + (k * np.log(n))
    pwlf_dic['bp1-tcrit'] = res2[1]
    pwlf_dic['bp1-slope1'] = my_pwlf2.calc_slopes()[0]
    pwlf_dic['bp1-slope2'] = my_pwlf2.calc_slopes()[1]
    pwlf_dic['bp1-bemis'] = my_pwlf2.predict(res2[1])[0]

    return pwlf_dic
#####################################
############ output Tcrit ###########
#####################################

def excluding_pandemic(city):
    # time_period = ("2020-01-01", india_data.index[-1])
    time_period = ("2020-01-01", india_data.index[-1])
    # time_period = (india_data.index[0], india_data.index[-1])
    # time_period = ("2020-01-01", '2021-12-31')

    df = extract_city(city)
    df = df[(df.index >= time_period[0]) & (df.index <= time_period[1])]["Total"]

    ERA5 = None
    if city not in cityname_map.keys():
        ERA5 = extract_city_ERA5(city)
        ERA5 = ERA5[(ERA5.index >= time_period[0]) & (ERA5.index <= time_period[1])]
    else:
        ERA5 = extract_city_ERA5(cityname_map[city])
        ERA5 = ERA5[(ERA5.index >= time_period[0]) & (ERA5.index <= time_period[1])]

    pwlf_dic = fit_piecewise_func(x=ERA5["mean"].values, y=df.values)

    return pwlf_dic['bp1-tcrit']

def including_pandemic(city):
    time_period = ("2019-01-01", india_data.index[-1])

    df = extract_city(city)
    df = df[(df.index >= time_period[0]) & (df.index <= time_period[1])]["Total"]

    ERA5 = None
    if city not in cityname_map.keys():
        ERA5 = extract_city_ERA5(city)
        ERA5 = ERA5[(ERA5.index >= time_period[0]) & (ERA5.index <= time_period[1])]
    else:
        ERA5 = extract_city_ERA5(cityname_map[city])
        ERA5 = ERA5[(ERA5.index >= time_period[0]) & (ERA5.index <= time_period[1])]

    pwlf_dic = fit_piecewise_func(x=ERA5["mean"].values, y=df.values)

    return pwlf_dic['bp1-tcrit']

def output_tcrit():
    ep_tcrit = []
    ip_tcrit = []

    for city in city_list:
        print(city, " Started")
        ep_tcrit.append(excluding_pandemic(city))
        ip_tcrit.append(including_pandemic(city))
        print(city, " Finished")

    # pd.DataFrame(data={"incl. pandemic": ip_tcrit, "excl. pandemic": ep_tcrit}, index=city_list).to_csv("./temperature_correlation/Tcrit.csv")

# ============================================
# 주요
# ============================================
globalVar = {}
serviceName = 'LSH0370'

# 옵션 설정
sysOpt = {
    # 시작/종료 시간
    # 'srtDate': '2019-01-01'
    # , 'endDate': '2023-01-01'
}

globalVar['inpPath'] = '/DATA/INPUT'
globalVar['outPath'] = '/DATA/OUTPUT'
globalVar['figPath'] = '/DATA/FIG'


#####################################
###### input data inventories #######
#####################################

inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'carbon-monitor-cities-India-v0325.csv')
fileList = sorted(glob.glob(inpFile))

# india_data = read_data.read()
india_data = pd.read_csv(fileList[0]).set_index("date")
india_data.index = pd.to_datetime(india_data.index)

# Index(['city', 'country', 'sector', 'value (KtCO2 per day)', 'timestamp'], dtype='object')
# india_data.columns


## ERA5 daily data
inpFile = '{}/{}/{}'.format(globalVar['inpPath'], serviceName, 'daily_temperature_india.csv')
fileList = sorted(glob.glob(inpFile))

# _ERA5 = pd.read_csv("./daily_temperature_india.csv").set_index("date")
_ERA5 = pd.read_csv(fileList[0]).set_index("date")
_ERA5.index = pd.to_datetime(_ERA5.index,format="%Y%m%d")

city_list=india_data["city"].unique().tolist()

## correspondence table for nearby cities (CMCC data vs ERA5 data)
## manually check in the websites below:
## https://mp.2markers.com/
## https://sz.far.city/

cityname_map={}
cityname_map["Ahmedabad"]="Gandhinagar" 
cityname_map["Ahmedabad"]      = "Gandhinagar"
cityname_map["Ambala"]         = "Patiala"
cityname_map["Asansol"]        = "Pashchim Medinipur"
cityname_map["Bengaluru"]      = "Kolar"
cityname_map["Canning"]        = "Haora"
cityname_map["Delhi"]          = "Gurgaon"
cityname_map["Dhing"]          = "Nagaon"
cityname_map["Ettumanoor"]     = "Kottayam"
cityname_map["Guruvayur"]      = "Thrissur"
cityname_map["Guwahati"]       = "Nagaon"
cityname_map["Haridwar"]       = "Dehradun"
cityname_map["Hyderabad"]       = "Mahbubnagar"
cityname_map["Imphal"]         = "Imphal west"
cityname_map["Kanpur"]         = "Kanpur Nagar"
cityname_map["Kharupetia"]     = "Udalguri"
cityname_map["Kochi"]          = "Kottayam"
cityname_map["Kolkata"]          = "Pashchim Medinipur"
cityname_map["Kuchai Kot"]     = "Gorakhpur"
cityname_map["Mangaluru"]      = "Kasaragod"
cityname_map["Mumbai"]         = "Thane"
cityname_map["Nibua Raiganj"]  = "Gorakhpur"
cityname_map["Piprakothi"]     = "Patna"
cityname_map["Prayagraj"]      = "Mirzapur"
cityname_map["Silvassa"]       = "Dadra and Nagar Haveli"
cityname_map["Tamluk"]         = "Haora"
cityname_map["Thalassery"]     = "Kozhikode"
cityname_map["Vijayawada"]     = "Guntur"

if __name__ == '__main__':
    #draw_plot("Suzhou (Ah)")
    output_tcrit()
