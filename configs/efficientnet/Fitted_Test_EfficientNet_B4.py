# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:17:34 2022

@author: MMClassification
"""
_base_ = [
    '../_base_/models/efficientnet_b4.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]




# ---- Model configs ----
# Here we use init_cfg to load pre-trained model.
# In this way, only the weights of backbone will be loaded.
# And modify the num_classes to match our dataset.

model = dict(
        init_cfg = dict(
            type='Pretrained', 
            checkpoint='/.../epoch_142.pth'             # Enter the epoch on which you want to do the test. Ex: epoch 142.
    ),
    head=dict(
        num_classes=5,                                  # Enter the same number of classes as in the train phase. Ex:5 in this case.
        topk = (1, ) 
    ))

workflow = [('train',1),('val',1)]                      # Depends on the specific machine

# dataset settings
dataset_type = 'CustomDataset'                          # The dataset is customized.

classes = ['AKIEC', 'BCC', 'KL', 'MEL', 'NV']   # Enter the exact class names, and enter them in the order they appear in the training, test, and validation sets.
                                                # Ex: in this case, those are the same names as the train phase classes, and in the same order.
                                                # In general, if we had 3 classes it would be: ['Class_1', 'Class_2', 'Class_3'].
#
# Data Pipeline: Make sure it is the same as the training phase
#
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

# ---- Dataset configs ----
data = dict(
    # Specify the batch size and number of workers in each GPU.
    # Please configure it according to your hardware.
    
    workers_per_gpu=10,                                           # Depends on the specific machine
    # Specify the training dataset type and path
    train=dict(
        type=dataset_type,
        data_prefix='/.../training_set/training_set',   # Enter the path of the training set.
        classes= classes,
        pipeline=train_pipeline),
    # Specify the validation dataset type and path
    val=dict(
        type=dataset_type,
        data_prefix='/.../val_set/val_set',             # Enter the path of the val set.
        ann_file= None ,
        classes= classes,
        pipeline=test_pipeline),
    # Specify the test dataset type and path
    test=dict(
        type=dataset_type,
        data_prefix='/.../test_set/test_set',           # Enter the path of the test set.
        ann_file= None ,
        classes= classes,
        pipeline=test_pipeline))
