import os
import glob
import json

use_folder = 'A'
dataset_root = os.getcwd()
dataset_dir = os.path.join(dataset_root, 'data')
train_dir = os.path.join(dataset_dir, 'train')


result = glob.glob(os.path.join(train_dir, use_folder,'*.jpg'))


class_name_list = []
for file_path in result:
    file_name = os.path.basename(file_path)
    class_name = file_name.split('_')[0]
    if class_name not in class_name_list:
        class_name_list.append(class_name)

dataset = {}
cnt = 0
for file_path in result:
    file_name = os.path.basename(file_path)
    class_name = file_name.split('_')[0]
    label = class_name_list.index(class_name)
    info = {
        'file_path' : file_path,
        'file_name' : file_name,
        'class_name' : class_name,
        'label' : label
    }
    dataset[cnt] = info
    cnt += 1 

with open(os.path.join(train_dir, 'annotation_' + use_folder + '.json'), 'w') as jsonfile: 
    json.dump(dataset, jsonfile)


