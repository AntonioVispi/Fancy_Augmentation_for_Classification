# Fancy_Augmentation_for_Classification
Tool to balance a dataset (unbalanced image classes), augment it in a classical way and then use it to train StyleGAN3. Once the StyleGAN3 has been trained, it will be used to generate synthetic images, which will allow to increase the available images, to then train a classifier coming from the MMClassification library (EfficientNET-B4), improving its final performance.

This work was done specifically for dermoscopic images (ISIC dataset for example). **However, one can use this Repository as a basis for performing the same complete procedure on completely different datasets, provided that simple changes are made to the image preprocessing:`Clean_Balancing.py`.**
Moreover, in the author's case, this method achieved better final classification accuracies than those obtained through classical augmentation methods.

Before discussing the steps, let's say that the repository is divided into two parts: the first dedicated to the reproduction of the result obtained with StyleGAN3, the second dedicated to the reproduction of the result obtained with MMClassification.


**If you need the synthetic images developed in this work, or are interested in trained models, both GAN and classification, please write to my mail:** antoniovispi1@gmail.com


# Getting started with GAN section

As a first step, we clone the repository, in order to define the first working environment, so then do the necessary installations. Alternatively you can run the GAN script in [Colab](https://colab.research.google.com/drive/1id8GZbrnR42xRQlkNukFr6UXtflAXCsc?usp=share_link).
```
git clone https://github.com/AntonioVispi/Fancy_Augmentation_for_Classification.git
cd Fancy_Augmentation_for_Classification
conda env create -f environment.yml
conda activate Tesi_Vispi_GAN
pip install -U albumentations
python -m pip install psutil
```
The entire first part, relating to StyleGAN3, can be performed within the first environment that has been defined.
We now begin the definition of the dataset.

(Optional step) If among the classes of your dataset, there is one class that is too numerous compared to the others, it is possible to select a subset of better images. The number of images you want will be selected. The selected images are simultaneously the heaviest, in terms of pixels, and the most resolute in terms of sharpness.

Enter the path of the too numerous class `path_in` and the path where you want the class with the desired number of images to be saved `path_out`. Also enter the number of images you want to keep. Ex: 5000.
```
python Select_Top_Images.py --path_in /.../NV --path_out /.../Top_NV --top_images_num 5000
```
We now move on to correcting and balancing the dataset. The function below will proceed to decrease the black annulus artifact typical of dermatological images from all the images that need it. Furthermore, this function allows to balance all the classes in terms of numbers, based on the most numerous class. The increases will be performed through classic operations (elastic distortions, flips, crops, rotations), coming from [Albumentation](https://github.com/albumentations-team/albumentations.git).

`path_in` is the path containing all the classes of the starting dataset. Note: make sure that only folders containing images of each class are in `path_in`.
`path_out` is the path where the new dataset will be saved, corrected and balanced.

```
python Clean_Balancing.py --path_in /.../Initial_Dataset \
--path_out /.../Dataset_Training
```
Now that the images have a more regular configuration, and are balanced in number, let's move on to assigning the labels.

In this phase the .json file containing the labels is created. Enter both as `input_folder` and as `output_folder` the same folder where the dataset obtained in the previous step is located. Ex: `/.../Dataset_Training`.
```
git clone https://github.com/JulianPinzaru/stylegan2-ada-pytorch-multiclass-labels.git
pip install Ninja
cd stylegan2-ada-pytorch-multiclass-labels
python make_json_labels.py --input_folder=/.../Dataset_Training --output_folder=/.../Dataset_Training
```
The result of the previous operation should be the following.
Before:
```bash
Dataset_Training
  │  
  ├── Class_1
  ├── Class_2
  ├── Class_3
  .
  .
  .  
  └── Class_n
```
After:
```bash
Dataset_Training
  │  
  ├── Class_1
  ├── Class_2
  ├── Class_3
  .
  .
  .
  ├── Class_n
  │   
  └── dataset.json
```
**Note: It is advisable to view the contents of the .json file obtained in this last step to note the correspondence between the labels and the classes. This annotation will be useful in the inference phase of the trained StyleGAN3. Ex: AKIEC corresponds to label 0, BCC corresponds to label 1 etc...**

Now that the dataset is fully defined, let's move on to the training phase of [StyleGAN3](https://github.com/NVlabs/stylegan3.git).

```
git clone https://github.com/NVlabs/stylegan3.git
cd stylegan3
```
Let's run the `dataset_tool.py`, which allows you to make all the images and all the labels in a format suitable for what StyleGAN3 expects. Adjust the desired resolution. In our case 1024x1024 pixels was used.
```
python dataset_tool.py --source /.../Dataset_Training --dest /.../Output_dataset_tool --resolution 1024x1024
```
In the path `/.../Output_dataset_tool` the final dataset to train the StyleGAN3 will be saved.

Now let's continue with the training of StyleGAN3 with `train.py`. The following block is used to start a new training from scratch.
For more information about the training parameters consult the source: [StyleGAN3](https://github.com/NVlabs/stylegan3.git).
```
python train.py --outdir /.../Output_train_StyleGAN3 \
--data /.../Output_dataset_tool \
--cfg=stylegan3-t --gpus=1 --batch=32 --gamma=32 --batch-gpu=4 --snap=5 --cond True --mirror=1
```
To resume a previous training, run the following block.
```
python train.py --outdir /.../Output_train_StyleGAN3 \
--data /.../Output_dataset_tool \
--cfg=stylegan3-t --gpus=1 --batch=32 --gamma=32 --batch-gpu=4 --snap=5 --cond True --mirror=1 \
--resume=/.../network-snapshot-000060.pkl
```
Once you finish training, it may be helpful to view the FID throughout your entire training session. To do this, the `FID_visualizer.py` function is provided.

So let's enter the `path_results` where all the training outputs are located (Following the example of this tutorial would be /.../Output_train_StyleGAN3), the `path_output` where we want the FID graph to be saved, and also the total number of images in the dataset. Ex: 25202.
```
cd Fancy_Augmentation_for_Classification
python FID_visualizer.py --path_results /.../Output_train_StyleGAN3 \
--path_output /.../FID_graph \
--dataset_images_num 25202
```
Make sure that only training outputs are in `path_results`, nothing else.
This function will display the lowest FID value and the corresponding epoch on the screen.

An example FID graph of a complete training from scratch is shown below.
![FID_Graph](https://user-images.githubusercontent.com/102518682/203628618-8aa4ab53-136b-423c-96b4-c3354203f0a5.jpg)

At this point it is possible to make the inference of the trained model as shown in the example below. 

The labels of the specific case are inside the .json file of the previous section expressed in bold.
You have to insert the label corresponding to the desired class in the entry: `class`.
`trunc` stands for truncation, by default it is set to 1. For more information consult [StyleGAN3](https://github.com/NVlabs/stylegan3.git). 

**Note: the following labels concern the two StyleGAN3 models trained in this work, with dataset images.**
```
example labels
#   AKIEC = 0   #
#   KL = 1      #
#   MEL = 2     #
#   NV = 3      #
#   BCC = 4     #
                           

cd stylegan3
python gen_images.py --outdir=/.../NV_fake \
--trunc=1 --seeds='1-5000' \                              # generation of 5000 images that will be saved in the folder /.../NV_fake
--network=/.../network-snapshot-000080.pkl --class=3      # in this example class=3 corresponds to the generation of NV images.

```
The following are examples of synthetic images belonging to some classes of the ISIC Dataset, generated by StyleGAN3, according to the procedure expressed up to this point.

Synthetic "Melanoma" (MEL class from ISIC):

![Norm_fake_Mel](https://user-images.githubusercontent.com/102518682/203645252-4ecc3917-1684-4b5d-9ae1-faa92daf527f.jpg)

Synthetic "Actinic Keratosis" (AKIEC class from ISIC):

![AKIEC](https://user-images.githubusercontent.com/102518682/233798660-ded1d0b3-a2e0-4704-a33a-2e07c0d8786d.jpg)

Synthetic "Basal Cell Carcinoma" (BCC class from ISIC):

![BCC](https://user-images.githubusercontent.com/102518682/233798675-729b312f-4240-4daa-a2c7-c9cdf1d4cad2.jpg)

Synthetic "Keratosis Like" (KL class from ISIC):

![KL](https://user-images.githubusercontent.com/102518682/233798734-15dbf962-3d8a-4d57-b527-ea1e6dff4852.jpg)

Synthetic "Nevus" (NV class from ISIC):

![NV](https://user-images.githubusercontent.com/102518682/233798762-cfe1f4d6-516d-430a-bd5c-9a2c9a5fec1f.jpg)





Before continuing, let's leave the environment of this first part to move on to the classification environment:
```
conda deactivate
```

# Getting started with Classification section

Before continuing with the discussion, let's define the environment within which the part relating to classification will be carried out.
Alternatively you can use the classifier train script on [Colab](https://colab.research.google.com/drive/1kd-P9O8BadD78FAUFRsvUx8xO1CSTxG_?usp=share_link).

```
conda create -n Tesi_Vispi_Classifier python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision==0.11.0 -c pytorch -y
conda activate Tesi_Vispi_Classifier
pip install openmim
pip install mmcv-full
git clone https://github.com/open-mmlab/mmclassification.git
cd mmclassification
pip install -e .
pip install mmcv -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install -U albumentations
pip install torchcam
```

To proceed with the use of fake images in a classification task, it is advisable to create a folder (Ex: `Dataset_Fake`), containing a sufficient number of fake images for each class, arranged, for example, as follows:
```bash
Dataset_Fake
  │  
  ├── Class_1
  ├── Class_2
  ├── Class_3
  .
  .
  .
  └── Class_n
```
Furthermore, we will also need the unbalanced dataset with real images, which will also need to be done this way:
```bash
Unbalanced_Dataset
  │  
  ├── Class_1
  ├── Class_2
  ├── Class_3
  .
  .
  .
  └── Class_n
```
**Be careful to keep the same class order between the fake dataset and the unbalanced dataset. Also class names must be identical between real and fake images.**

Now we need to define the dataset that will be given to the classifier.

Two different modes have been made available. The first method of defining the dataset consists in balancing the classes, through traditional increases (elastic distortions, flips, crops, rotations), based on the most numerous class. Talking about the function `Classic_Balanced_Classifier_Dataset.py`.
The other method consists in bridging the imbalance of the classes through the Fake images, generated in the previous step: `Dataset_Fake`. Then, through the function `Fake_Balanced_Classifier_Dataset.py` we will proceed with the creation of the dataset balanced with Fake images for the classifier.

So, let's move on `Fancy_Augmentation_for_Classification`.
```
cd Fancy_Augmentation_for_Classification
```
To balance the dataset with classical methods, run the following block:
```
python Classic_Balanced_Classifier_Dataset.py --path_in /.../Unbalanced_Dataset \
--path_out /.../Classifier_Dataset
```
To balance the dataset with Fake images, run the following block:
```
python Fake_Balanced_Classifier_Dataset.py --path_in /.../Unbalanced_Dataset \
--path_in_fake /.../Dataset_Fake \
--path_out /.../Classifier_Dataset
```
At this point, regardless of how the dataset has been balanced, the final dataset that will be used to train the classifier will look exactly like this (Suppose we have only 3 classes):
```bash
Classifier_Dataset
  │  
  ├── test_set
  │          └── test_set
  │                     ├── Class_1
  │                     ├── Class_2
  │                     └── Class_3
  ├── training_set
  │          └── training_set
  │                     ├── Class_1
  │                     ├── Class_2
  │                     └── Class_3
  └── val_set
             └── val_set
                        ├── Class_1
                        ├── Class_2
                        └── Class_3                        
 
```
For example, inside `Class_1` there will be images related to `Class_1` and so on.
Note: `Classic_Balanced_Classifier_Dataset.py` will generate two folders in the desired path, one relating to the balanced dataset, another relating to the complete dataset for the classifier; while `Fake_Balanced_Classifier_Dataset.py` will directly generate the complete dataset for the classifier, without any other folders.

Now that the dataset(s) are defined, we move on to the training phase. 

To do this, a useful training function is provided: `Train_Classifier_Fitted.py`.

If you want to replicate the results of this work, you have to use `Train_Classifier_Fitted.py`. Otherwise, you can modify the training parameters of the aforementioned function as desired. But, keep in mind that any changes you make will have to be extended to the test script as well.


Note that `Train_Classifier_Fitted.py` is not a command line, so it is recommended to use it offline, in order to be able to insert all the eventual settings necessary for the specific case of the user.

For more information about the training settings, consult the source: [MMClassification](https://github.com/open-mmlab/mmclassification.git).

Once the training is finished, it could be useful to view the trend of the accuracies and losses, on the training set and on the validation set,  for example to understand if overfitting has occurred. To do this, the `show_classifier_graphs.py` function was implemented. 
In order to use it, it is necessary to go inside the training output folder; there will be a `.json` file containing the training data. Use the latter as input to the `show_classifier_graphs.py` function. While in the output folder the image will be saved with the name of `Graphs.png`

Thus, to see the training graphs, type:
```
cd Fancy_Augmentation_for_Classification
python show_classifier_graphs.py --path_in_json /.../example_20221123_093800.log.json \
--path_out /.../Your_Output_Directory
```
If you want to test this feature, a `.json` file is provided in this repository in the folder `Fancy_Augmentation_for_Classification/Demo/20221116_191229.log.json`.

Below is an example of the graphs obtained during a training of the EfficientNet-B4 with the settings used in `Train_Classifier_Fitted.py`.
![Train_Classifier_Graphs](https://user-images.githubusercontent.com/102518682/204105299-9aef5d8e-6ffe-4234-a383-5810c43094ca.jpg)

## Testing the model
We continue with the test of the trained model. Alternatively the test script available on Colab is [here](https://colab.research.google.com/drive/1BU1TUPkeC865b8TAS7g5eOXY-3WePD5r?usp=share_link).

 
**`Fancy_Augmentation_for_Classification/configs/efficientnet/Fitted_Test_EfficientNet_B4.py` can be used to test the newly trained model with the same training setup as this work.**

If changes have been made to the train settings, with respect to those proposed in this repository, it is crucial to make the same changes to `Fitted_Test_EfficientNet_B4.py`.
Particular attention is paid to the number of classes, the name of the classes, the Data Pipeline, the Dataset configs: make sure they are like those set in the train phase(If any changes were made to it).

In any case, you have to insert in to `Fitted_Test_EfficientNet_B4.py` the path corresponding to the checkpoint on which you want to test the model (file `.pth`, in line with the entry checkpoint), and the paths of your dataset: the same inserted in the training phase, at training, val and test sets.

**Example: Suppose we want to test the model, trained as explained above. I don't need to make any changes to the `Fitted_Test_EfficientNet_B4.py` file, other than inserting the dataset paths, and the corresponding trained model path.**

Therefore, once the right paths have been entered, in the configuration(`Fitted_Test_EfficientNet_B4.py`) for the dataset we want to test, we can proceed.

Assuming the right changes, as explained above, have been made, to `Fitted_Test_EfficientNet_B4.py`, run the following block to calculate the average accuracy top-1 across all test images (Ex: with the checkpoint saved at epoch 142):
```
cd mmclassification
python tools/test.py /.../Fancy_Augmentation_for_Classification/configs/efficientnet/Fitted_Test_EfficientNet_B4.py \
/.../epoch_142.pth \
--metrics accuracy --metric-options topk=1
```
For generation , and saving in the desired output folder, a `.pkl` file that contains all the details of the results on the images of the Test Set:
```
python tools/test.py /.../Fancy_Augmentation_for_Classification/configs/efficientnet/Fitted_Test_EfficientNet_B4.py \
/.../epoch_142.pth \
--out /.../your_output_path/Results.pkl --out-items class_scores all
```
For calculate the Confusion Matrix on the Test Set, using the previously generated `.pkl` file, please run:
```
import mmcv
from mmcls.datasets import build_dataset
from mmcls.core.evaluation import calculate_confusion_matrix
cfg = mmcv.Config.fromfile('/.../Fancy_Augmentation_for_Classification/configs/efficientnet/Fitted_Test_EfficientNet_B4.py')
dataset = build_dataset(cfg.data.test)
pred = mmcv.load('/.../your_output_path/Results.pkl')['class_scores']
matrix = calculate_confusion_matrix(pred, dataset.get_gt_labels())
print(matrix)

import matplotlib.pyplot as plt
plt.imshow(matrix)   
plt.show()
```
This is an example of confusion matrix obtained from the procedure described up to now:

![conf_matrix](https://user-images.githubusercontent.com/102518682/204387381-a1e8b058-4df7-4844-bd08-b5a0b3e48fcb.png)

Finally: Precision, Recall, F1 Score.
Specify if you are interested in the values of the individual classes: average_mode = 'none'.
Or to average values across all classes: average_mode = 'macro'. Please, run:
```
from mmcls.core.evaluation import precision_recall_f1
precision_recall_f1(pred, dataset.get_gt_labels(),average_mode='none')
```
## Inference & Saliency map

For viewing the inference and saliency map, again, it is recommended to use `Inference_and_Saliency_map.py`  offline, in order to be able to enter the settings that suit your case, such as the specific classes you have. Otherwise you can use the [Colab](https://colab.research.google.com/drive/1MsucdDeHFre1JdPbrP3BbT-PXxHBBFuv?usp=share_link) script.

To generate the saliency map and the inference, it is advisable to use (at `checkpoint_file` which is inside `Inference_and_Saliency_map.py`) the best network trained in this work: the one related to the dataset balanced and augmented with only the synthetic images generated as described above in this work.

If you follow the instructions contained in `Inference_and_Saliency_map.py`, adapted to your case, the result will be, for example, the following:

![Mel_Norm_4](https://user-images.githubusercontent.com/102518682/204402509-a072bc59-fceb-4842-9635-732c809ffe55.png)

At this point, you can exit this environment as well.
```
conda deactivate
```










