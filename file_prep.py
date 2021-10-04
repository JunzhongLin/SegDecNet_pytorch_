import os
import glob
import shutil

raw_path = r'./raw_data/KolektorSDD'
work_path = r'./Data'

raw_img_file_list = glob.glob(os.path.join(raw_path, 'kos*/*.jpg'))
raw_mask_file_list = glob.glob(os.path.join(raw_path, 'kos*/*.bmp'))
train_ok_mask_list = glob.glob(os.path.join(work_path, 'Train_OK/*.bmp'))
train_ng_mask_list = glob.glob(os.path.join(work_path, 'Train_NG/*.bmp'))

raw_img_file_list.sort()
raw_mask_file_list.sort()
train_ok_mask_list.sort()
train_ng_mask_list.sort()

on_disk_raw_img = []
on_disk_raw_mask = []

# Prepare the training set

# for item_file in train_ng_mask_list:
#     name = item_file.split('/')[-1]
#     raw_file_path = os.path.join(raw_path, name.split('_')[0], name.split('_')[1]+'.jpg')
#     target_file_path = os.path.join(work_path, 'Train_NG', name.split('_')[0]+'_'+name.split('_')[1]+'.jpg')
#     on_disk_raw_img.append(target_file_path)
#     if raw_file_path in raw_img_file_list:
#         shutil.copy(src=raw_file_path, dst=target_file_path)

# prepare the test set

in_train_img_list = []
for item_file in train_ng_mask_list+train_ok_mask_list:
    name = item_file.split('/')[-1]
    raw_file_path = os.path.join(raw_path, name.split('_')[0], name.split('_')[1] + '.jpg')
    in_train_img_list.append(raw_file_path)

for item_file in raw_img_file_list:
    if item_file not in in_train_img_list:
        temp = item_file.split('/')
        target_file_path = os.path.join(work_path, 'Test', temp[-2]+'_'+temp[-1])
        shutil.copy(src=item_file, dst=target_file_path)
        shutil.copy(src=item_file[:-4]+'_label.bmp', dst=target_file_path[:-4]+'_label.bmp')


