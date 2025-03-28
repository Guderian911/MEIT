# Mixture of Expers Image Transfomer For Bitemporal Remote Sensing Images Change Detection

Here, we provide the pytorch implementation of the paper: Mixture of Expers Image Transfomer For Bitemporal Remote Sensing Images Change Detection.
Core code is placed in model/MEIT.

![image](./images/img.png)


## Requirements

```
Python 3.12
pytorch 2.3.0
torchvision 0.18.0
einops  0.8.0
```

## Installation

Clone this repo:

```shell
git clone https://github.com/Guderian911/MEIT.git
cd MEIT
```

## Quick Start

We have some samples from the [LEVIR-CD](https://justchenhao.github.io/LEVIR/) dataset in the `samples` folder for a quick start.

Firstly, you can download our BIT pretrained model——by [baidu drive, code: MEIT](https://pan.baidu.com/s/1du6BWp_nbffwekbUAUk0IQ). After downloaded the pretrained model, you can put it in `checkpoints`.

Then, run a demo to get started as follows:

```python
python demo.py 
```

After that, you can find the prediction results in `samples/predict`.

## Train

You can find the training script `run_cd.sh` in the folder `scripts`. You can run the script file by `sh scripts/run_cd.sh` in the command environment.

The detailed script file `run_cd.sh` is as follows:

```cmd
gpus=0
checkpoint_root=checkpoints 
data_name=train  # dataset name 

img_size=512
batch_size=4
lr=0.01
max_epochs=200  #training epochs
net_G=MEIT # model name
lr_policy=linear

split=train  # training txt
split_val=val  #validation txt
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}
```

## Evaluate

You can find the evaluation script `eval.sh` in the folder `scripts`. You can run the script file by `sh scripts/eval.sh` in the command environment.

The detailed script file `eval.sh` is as follows:

```cmd
gpus=0
data_name=LEVIR # dataset name
net_G=MEIT # model name 
split=test # test.txt
project_name=MEIT_LEVIR # the name of the subfolder in the checkpoints folder 
checkpoint_name=best_ckpt.pt # the name of evaluated model file 

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}
```

## Dataset Preparation

### Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.

### Data Download 

LEVIR-CD: https://justchenhao.github.io/LEVIR/

WHU-CD: https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html

DSIFN-CD: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset

## License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

## Citation

If you use this code for your research, please cite our paper:

```
@Article{chen2021a,
    title={Remote Sensing Image Change Detection with Transformers},
    author={Hao Chen, Zipeng Qi and Zhenwei Shi},
    year={2021},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    volume={},
    number={},
    pages={1-14},
    doi={10.1109/TGRS.2021.3095166}
}
```
