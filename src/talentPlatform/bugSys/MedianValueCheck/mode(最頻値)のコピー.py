import statistics
import open3d as o3d
import numpy as np
import glob
from natsort import natsorted
import pandas as pd
import os

# set depth range: mm
min_depth = 0
max_depth = 1000
# depth_scale of L515
depth_scale = 0.0002500000118743628

# 関心領域(ROI): pixel
roi_x_min = 0
roi_x_max = 640
roi_y_min = 0
roi_y_max = 480

# Your color_img folder and depth_img folder
# # depth_img_folder = "/Volumes/My Passport/20211010_0831/realsense/depth"
depth_img_folder = 'E:/04. TalentPlatform/Github/TalentPlatform-Python/src/talentPlatform/bugSys/MedianValueCheck/depth'

# Get path(.jpg and .png) in a folder
depth_path = natsorted(glob.glob(depth_img_folder + "/*.png"))

time_list = []
data_list = []
#print(depth_path)
#for i in range(len(depth_path)):
for i in range(10):

    if i >= len(depth_path): continue

    # read a depth image
    root_ext = os.path.splitext(depth_path[i])
    root_split = os.path.split(root_ext[0])
    file_name = root_split[1]
    time = str(file_name[:4]) + "-" + str(file_name[4:6]) + "-" + str(file_name[6:8]) + " " + \
           str(file_name[9:11]) + ":" + str(file_name[11:13]) + ":" + str(file_name[13:15]) + "." + str(
        file_name[15:])

    depth_raw = o3d.io.read_image(depth_path[i])
    depth_img = np.asanyarray(depth_raw) * depth_scale * 1000  # : mm

    # take a depth out of ROI
    depth_list = depth_img[roi_y_min:roi_y_max + 1, roi_x_min:roi_x_max + 1]
    depth_list = np.ravel(depth_list)

    # 範囲内のdepthを平均処理
    distance = [i for i in depth_list if min_depth < i < max_depth]
    # depthList = [i for i in depth_list if min_depth < i < max_depth]

    # if len(distance) < 1: continue
    if len(distance) < 1: distance = [0]

    distance = np.round(distance, decimals=1)

    mode = statistics.mode(distance)
    print(mode)

    time_list.append(time)
    data_list.append(mode)

    df = pd.DataFrame({"time": time_list, "depth": data_list})

    # df.to_csv("/Users/jeungyoungjun/Desktop/20211010_otani_秋耕耘/RealSense/20211010_otani_26-1_mode.csv", index=False)

