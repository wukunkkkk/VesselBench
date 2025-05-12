import json
from os.path import join
import glob


if __name__ == '__main__':
    # path to folder that contains images
    
    img_folder_train = './vessel_dataset_for_CAN/train_data/images'
    img_folder_val = './vessel_dataset_for_CAN/val_data/images'
    img_folder_test = './vessel_dataset_for_CAN/test_data/images'
    
    # path to the final json file
    output_json_train = 'train.json'
    output_json_val = 'val.json'
    output_json_test = 'test.json'
    
    img_list = []

    for img_path in glob.glob(join(img_folder_train,'*.tif')):
        img_list.append(img_path)
    with open(output_json_train,'w') as f:
        json.dump(img_list,f)
        
    for img_path in glob.glob(join(img_folder_val,'*.tif')):
        img_list.append(img_path)
    with open(output_json_val,'w') as f:
        json.dump(img_list,f)
        
    for img_path in glob.glob(join(img_folder_test,'*.tif')):
        img_list.append(img_path)
    with open(output_json_test,'w') as f:
        json.dump(img_list,f)
