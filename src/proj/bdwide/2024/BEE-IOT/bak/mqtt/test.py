import os

import traceback
import sys

data_path = '/HDD/DATA/BEE-IOT/CSV/'    
# data_path = '/home/dms01user01/iot_server/mqtt/'
file_name = 'test.txt'

CSV_FILE = data_path + file_name
# CSV_FILE = file_name

# with open(CSV_FILE, mode='w', newline='') as csv_file:
#         csv_file.write("testtesttest")

with open(CSV_FILE, mode='a', newline='') as csv_file:
        csv_file.write("test")