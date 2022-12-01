"""
Created on Sat Nov 26 17:40:32 2022
@author: MMClassification
"""

"""
Automatically generated by Colaboratory.
"""

"""
NOTE: The only data input and output paths are the paths where are located train, val and test set, 
and the saving path of the training exits.
The train val and test paths are double, in the sense that for example there is a train
set inside the tain set; you have to take the deeper one
(therefore the second in order of opening), the same goes for val and test set.
For the labels, please enter the same class names, with the same order appearing in both 
train and val and test set, as shown in the following code. Ex: classes =['Class_1','Class_2','Class_3']
Finally you need to enter the path of the output folder.
    
"""


# Move in MMClassification
cd mmclassification



# Check MMClassification installation
import mmcls
print(mmcls.__version__)

""" 
Specify the model configuration from MMClassification
"""

# Confirm the config file exists
!ls /.../mmclassification/configs/efficientnet/efficientnet-b4_8xb32_in1k.py

# Specify the path of the config file.
config_file = '/.../mmclassification/configs/efficientnet/efficientnet-b4_8xb32_in1k.py'  # Look for the configuration corresponding to the EfficientNet-B4

import mmcv
from mmcls.apis import inference_model, init_model, show_result_pyplot

"""
Complete preparation of training setup
In this example, a dataset with 5 classes from ISIC was used.
In this phase, the training and data processing settings are customized.
"""

# Load the base config file
from mmcv import Config
cfg = Config.fromfile(config_file)

# Modify the number of classes in the head.
cfg.model.head.num_classes = 5                     # AKIEC , BCC , KL , MEL , NV
cfg.model.head.topk = (1, )

cfg.model.head.cal_acc = True     

cfg.device='cuda'    # GPU

cfg.workflow = [('train',1),('val',1)]   
################################################ 

cfg.runner = dict(type='EpochBasedRunner', max_epochs=150)                                    # n° epochs
cfg.lr_config = dict(policy='step', step=[30, 45, 68, 110, 140, 145], gamma=0.316227766)      # Learning rate sheduler

################################################ 
# dataset settings
dataset_type = 'CustomDataset'

classes = ['AKIEC', 'BCC', 'KL', 'MEL', 'NV']   #Enter the exact class names, and enter them in the order they appear in the training, test, and validation sets.

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=380,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CenterCrop',
        crop_size=380,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# Specify sample size and number of workers.
cfg.data.workers_per_gpu = 10                 # This depends on the specific machine

# Specify the path and meta files of training dataset
cfg.data.train.data_prefix = '/.../training_set/training_set'
cfg.data.train.classes = classes

# Specify the path and meta files of validation dataset
cfg.data.val.data_prefix = '/.../val_set/val_set'
cfg.data.val.ann_file = None
cfg.data.val.classes = classes

# Specify the path and meta files of test dataset
cfg.data.test.data_prefix = '/.../test_set/test_set'
cfg.data.test.ann_file = None
cfg.data.test.classes = classes



# Specify the output work directory
cfg.work_dir = '/.../Your_Output_Directory'


# Config to set the checkpoint hook
cfg.checkpoint_config.interval = 1   # The save interval is 1 epoch


# Set the random seed and enable the deterministic option of cuDNN
# to keep the results' reproducible.

from mmcls.apis import set_random_seed
cfg.seed = 0
set_random_seed(0, deterministic=True)

cfg.gpu_ids = range(1)

"""
Classifier training.
Make sure you have a large GPU. The A100 is recommended.
"""
#########################
#   Train the model     #
#########################
import time
import mmcv
import os.path as osp

from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.apis import train_model

# Create the work directory
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# Build the classifier
model = build_classifier(cfg.model)
model.init_weights()
# Build the dataset
datasets = [build_dataset(cfg.data.train)]

###################################
import copy
val_dataset = copy.deepcopy(cfg.data.val)
val_dataset.pipeline = cfg.data.train.pipeline
datasets.append(build_dataset(val_dataset))

###################################

# Add `CLASSES` attributes to help visualization
model.CLASSES = datasets[0].CLASSES
# Start fine-tuning
train_model(
    model,
    datasets,
    cfg,
    distributed=False,
    validate=True,
    timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
    meta=dict())



















