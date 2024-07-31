from wwo_hist import retrieve_hist_data
import os

with open('states.txt', 'r') as f:
    states = f.readlines()
    BASE_PATH = os.getcwd()
    os.chdir('data')
    downloaded_states = [state.split('.')[0] for state in os.listdir()]
    downloaded_states.append(['Indiana','Kansas','New+Hampsphire'])
    os.chdir(BASE_PATH)
    print(downloaded_states)
    location_list = [state.split(',')[1].strip() for state in states if state.split(',')[1].strip() not in downloaded_states]

import os
os.chdir("./data")


frequency = 24
start_date = '25-JUNE-2011'
end_date = '25-JUNE-2021'
api_key = ''
# api_key = '392e2b4295b04f38a9f14846212906'

hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)