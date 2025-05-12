# VesselBench

This project provides executable code for tasks such as object detection and counting on our VesselBench-800K dataset.

## Object Detection

For object detection tasks, we use [MMDetection](https://github.com/open-mmlab/mmdetection) as the baseline framework. Please follow the official MMDetection repository for `installation` and `environment setup`. In this project, we provide configuration files for two representative models: Faster R-CNN and TOOD. You can directly use these configs with MMDetection to train and evaluate on the VesselBench-800K dataset.

The configuration files are located in the `my_configs` directory. Please make sure to modify the `data_root` in the configuration `faster_rcnn,py` files according to your local dataset location before use.

### Train a new model
Please place the my_configs folder into the root directory of your local MMDetection installation.
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
