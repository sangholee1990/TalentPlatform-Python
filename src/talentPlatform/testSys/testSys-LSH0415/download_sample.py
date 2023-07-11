# Demo 1-1. Setting an enviroment (download_sample.py)
from os import listdir, mkdir, path, system, getcwd
import warnings; warnings.simplefilter("ignore")
dir_origin = dir_origin = getcwd()+'/' # <- Change this in local machine
dir_dataset= 'dataset/'
print('\n1)============ Start Downloading =================\n')
print('Target directory ... => [%s%s]'%(dir_origin,dir_dataset))

#!rm -rf /content/dataset/
import requests, time
def download_dataset( animal_list = range(1,7), dir_dataset = dir_dataset ):
  # Check directory
  if not path.isdir('%s%s'%(dir_origin,dir_dataset)):
    mkdir('%s%s'%(dir_origin,dir_dataset))
    mkdir('%s%s/dataset_1/'%(dir_origin,dir_dataset))
    mkdir('%s%s/dataset_2/'%(dir_origin,dir_dataset))

  # File names to be downloaded
  file_ids = [ 'meta.csv', 'montage.csv' ]
  for set_id in animal_list:
    file_ids.append( 'dataset_1/epochs_animal%s.set'%set_id )
    file_ids.append( 'dataset_1/epochs_animal%s.fdt'%set_id )
    file_ids.append( 'dataset_2/epochs_animal%s.set'%set_id )
    file_ids.append( 'dataset_2/epochs_animal%s.fdt'%set_id )

  # Request & download
  repo_url = 'https://gin.g-node.org/hiobeen/Mouse_hdEEG_ASSR_Hwang_et_al/raw/b986d72322caa0ec76f02211dcc5c1df45d8366e/'
  for file_id in file_ids:
    fname_dest = "%s%s%s"%(dir_origin, dir_dataset, file_id)
    if path.isfile(fname_dest) is False:
      print('...copying to [%s]...'%fname_dest)
      file_url = '%s%s'%(repo_url, file_id)
      r = requests.get(file_url, stream = True)
      with open(fname_dest, "wb") as file:
          for block in r.iter_content(chunk_size=1024):
              if block: file.write(block)
      time.sleep(1) # wait a second to prevent possible errors
    else:
      print('...skipping already existing file [%s]...'%fname_dest)

# Initiate downloading
animal_list = [2] # Partial download to prevent long download time
#animal_list = [1,2,3,4,5,6] # Full download 
download_dataset(animal_list)
print('\n============= Download finished ==================\n\n')

# List up 'dataset/' directory
print('\n2)=== List of available files in google drive ====\n')
print(listdir('%sdataset/'%dir_origin))
print('\n============= End of the list ==================\n\n')

# Install mne-python module
system('pip install mne');

# Make figure output directory
dir_fig = 'figures/'
if not path.isdir(dir_fig): mkdir('%s%s'%(dir_origin, dir_fig))
  
