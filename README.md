# VesselBench-800K

This repository provides executable code for tasks like `object detection` and `density estimation` on our VesselBench-800K dataset.

## Object Detection

For object detection tasks, we use **[MMDetection](https://github.com/open-mmlab/mmdetection)** as the baseline framework. Please follow the official MMDetection repository for `installation` and `environment setup`. In this project, we provide configuration files for two representative models: Faster R-CNN and TOOD. You can directly use these configs with MMDetection to **train and evaluate** on the **VesselBench-800K** dataset.

### Dataset directory structure
After downloading and unzipping the `dataset from Kaggle`, ensure that the VesselBench directory structure is as follows:
```bash
VesselBench/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
│   └── instances_test.json
├── train-msi/
│   ├── *.tif
│   └── ...
├── train-sar/
│   ├── *.tif
│   └── ...
├── train-dot/
│   ├── *.png
│   └── ...
├── val-msi/
├── val-sar/
├── val-dot/
├── test-msi/
├── test-sari/
├── test-dot/
```

The configuration files are located in the `my_configs` directory. Please put the `my_configs` folder into the root directory of your local MMDetection installation and make sure to modify the `data_root` in the configuration files according to your local dataset location before use.
```bash
MMDetection/
├── configs/
├── mmdet/
├── tools/
├── my_configs/
│   ├── faster_rcnn_vesselbench.py
│   └── tood_vesselbench.py
├── requirements.txt
├── ...
```

### Train a new model
```bash
cd 'your MMdetection path'
conda activate 'your environment'
```
To train a model with the new config, you can simply run
```bash
python tools/train.py my_configs/faster-rcnn.py
```
or 
```bash
python tools/train.py my_configs/tood.py
```

### Test and inference
To test the trained model, you can simply run
```bash
python tools/test.py my_configs/faster-rcnn.py work_dirs/`checkpoint file`
```

## Density Estimation

For density estimation or counting tasks, we provide the **[Context-Aware Crowd Counting](https://github.com/weizheliu/Context-Aware-Crowd-Counting)** method as an example implementation. Please refer to the original repository for environment setup instructions. You can follow the steps below to apply this method to our **VesselBench-800K** dataset.

**Please download the `CAN_vesselbench` folder directly from this repository.**

### Data Preparation 
Since ground truths used for density estimation are not directly compatible with those used for object detection, different density estimation methods often require generating different formats of GT files — and the CAN method is no exception. **This process can be tedious.**

**Therefore, we provide a subset of the full VesselBench dataset on Kaggle, specifically prepared and formatted for direct use with the CAN method. You can download and extract it into the `CAN_vesselbench` folder.**

After extraction, make sure the folder structure is as follows:

```bash
CAN_vesselbench/
├── vessel_dataset_for_CAN/
│   ├── train_data/
│   │   └── images/
│   │   └── ground_truths/
│   ├── val_data/
│   ├── test_data/
├── ...
├── train.py
├── test.py
├── ...
```

Then, use `create_json.py` to generate the json file `train.json` `val.json` and `test.json` which contains the path to the images.
```bash
python create_json.py
```


### Training
In command line:
```bash
python train.py train.json val.json
```

### Testing
1.  Modify the "test.py", make sure the path is correct.

2.  In command line:

```bash
python test.py
```



