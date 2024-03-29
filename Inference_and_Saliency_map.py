"""
Inference & Saliency Map

 Automatically generated by Colaboratory.

"""


# Move in MMClassification
cd mmclassification


# Check MMClassification installation
import mmcls
print(mmcls.__version__)
from mmcls.apis import inference_model, init_model, show_result_pyplot

"""
# Insert the model configuration (MMClassification) in config_file
# Insert in checkpoint_file the .pth weights of the desired network

"""

# Confirm the config file exists
ls /.../mmclassification/configs/efficientnet/efficientnet-b4_8xb32_in1k.py

# Specify the path of the config file and checkpoint file.

config_file = '/.../mmclassification/configs/efficientnet/efficientnet-b4_8xb32_in1k.py'
checkpoint_file = '/.../epoch_137.pth'

"""
I replicate all the configurations of the trained model

"""

# Load the base config file
from mmcv import Config
from mmcls.utils import auto_select_device

cfg = Config.fromfile(config_file)

device='cpu'

"""
In this example we have exactly these 5 classes.

"""

classes = ['AKIEC', 'BCC', 'KL', 'MEL', 'NV']   #Enter the exact class names, and enter them in the order they appear in the training, test, and validation sets.

# Load the pre-trained model's checkpoint.
cfg.model.init_cfg = dict(type='Pretrained', checkpoint=checkpoint_file)

# Modify the number of classes in the head.
cfg.model.head.num_classes = 5                      # we have 5 classes
cfg.model.head.topk = (1, )

# Specify the path and meta files of training dataset
cfg.data.train.classes = classes

# Specify the path and meta files of validation dataset
cfg.data.val.ann_file = None
cfg.data.val.classes = classes

# Specify the path and meta files of test dataset
cfg.data.test.ann_file = None
cfg.data.test.classes = classes

# Model activation

model = init_model(cfg, checkpoint_file, device)  # model initialization

""" 
I define the function that takes as input the image whose saliency map 
is desired, to transform it into the same format that the network expects
"""

from mmcv.parallel import collate, scatter
from mmcls.datasets.pipelines import Compose
import numpy as np

def image_fixer(model, img):

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    return data['img']

"""
Insert the desired image specifying the real class it belongs to. If the real class is not known, specify 'Unknown'.
"""

########################### enter your image

# enter the full path of the image
immagine = ('/.../Example_ISIC_0000029.png') 

# Enter the true class of the image, if known. (Ex: AKIEC, BCC, KL, MEL, NV)
classe_vera = 'MEL'                         

########################### enter your image









""" 
Creation of image format suitable for the model
"""

img = image_fixer(model, immagine)

"""
Generation Saliency Map
"""

from PIL import Image
from argparse import ArgumentParser
import torch
from mmcls.apis import inference_model, init_model, show_result_pyplot

img.requires_grad = True
y = model(img, return_loss=False,  post_process=False)
print(type(y), y.size())
score, indices = torch.max(y, 1)
score.backward()
slc, _ = torch.max(torch.abs(img.grad[0]), dim=0)
#normalize to [0..1]
slc = (slc - slc.min())/(slc.max()-slc.min())
print(slc.size())

"""# Inference of the model"""

model.cfg = cfg
result = inference_model(model, immagine)
result['Real_class'] = classe_vera 

# Placing the inference output data into a string for final display in the graph

pred_label = str(result['pred_label'])
pred_score = str(round(result['pred_score'],2))   
pred_class = str(result['pred_class'])
real_class = str(result['Real_class'])

s ='Pred_label: %(pred_label)s \nPred_score: %(pred_score)s \nPred_class: %(pred_class)s\nReal_class: %(real_class)s' % { "pred_label": pred_label, "pred_score": pred_score, "pred_class": pred_class, "real_class": real_class}

""" 
Graph View with Inference & Saliency Map
"""

import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image
import scipy.ndimage as ndimage



img_vis = Image.open(immagine)

#define transforms to preprocess input image into format expected by model

# transform for square resize
transform = T.Compose([
    T.Resize((380,380))
])
resized_img = transform(img_vis)

################

saliency=slc.numpy()
immagine_plot = np.array(resized_img)
################
img = saliency*15                 # I increase the values of the saliency map to facilitate the visualization

img = ndimage.gaussian_filter(img, sigma=10, order=0)
img = ndimage.gaussian_filter(img, sigma=10, order=0)  # Lowpass of the saliency map, to beautify the display
img = ndimage.gaussian_filter(img, sigma=10, order=0)


result = overlay_mask(to_pil_image(immagine_plot), to_pil_image(img, mode='F'), alpha=0.5, colormap = 'jet')

################
#plot image and its saleincy map

plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
plt.imshow(resized_img)
plt.text(10, 60, s, fontsize=15, color='white', bbox=dict(fill=True, color ='black'))
plt.xticks([])
plt.yticks([])
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.xticks([])
plt.yticks([])
plt.show()
