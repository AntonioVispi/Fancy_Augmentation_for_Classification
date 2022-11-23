# Fancy_Augmentation_and_Classification
Tool for creating a balanced dataset, augmenting it in the classic way, and then using it to train StyleGAN3. Once the StyleGAN 3 has been trained, it will be used to generate synthetic images, which will allow to increase the images available to train an MMClassification classifier (EfficientNET-B4).

Before discussing the steps, let's say that the repository is divided into two parts: the first dedicated to the reproduction of the result obtained with StyleGAN3, the second dedicated to the reproduction of the result with MMClassification.

# Getting started with GAN section

The first phase is the creation of the dataset. please install:
```
git clone https://github.com/AntonioVispi/Fancy_Augmentation_for_Classification.git
cd Fancy_Augmentation_and_Classification
pip install -U albumentations
```
(Optional step) If among the classes of your dataset, there is one class that is too numerous compared to the others, it is possible to select a subset of better images. The number of images you want will be selected. The selected images have the best resolution of all.

Enter the path of the too numerous class `path_in` and the path where you want the class with the desired number to be saved `path_out`. Also enter the number of images you want. Ex: 5000.
```
python Select_Top_Images.py --path_in /.../NV --path_out /.../Top_NV --top_images_num 5000
```
We now move on to correcting and balancing the dataset. The function below will proceed to decrease the black annulus artifact typical of dermatological images from all images. Furthermore, this function allows to balance all the classes in terms of numbers, based on the most numerous class. The increases will be performed through classic operations (elastic distortions), coming from Albumentation.

`path_in` is the path containing all the classes of the starting dataset. Note: make sure that only folders containing images of each class are in `path_in`.
`path_out` is the path where all the dataset will be saved, corrected and balanced.

```
!python Clean_Balancing.py --path_in /.../Initial_Dataset \
--path_out /.../Dataset_Training
```
Now that the images are correct and balanced in number, let's move on to assigning the labels.

In this phase the .json file containing the labels is created. Enter both as `input_folder` and as `output_folder` the same folder where the dataset obtained in the previous step is located. Ex: `/.../Dataset_Training`.
```
git clone https://github.com/JulianPinzaru/stylegan2-ada-pytorch-multiclass-labels.git
pip install Ninja
cd stylegan2-ada-pytorch-multiclass-labels
python make_json_labels.py --input_folder=/.../Dataset_Training --output_folder=/.../Dataset_Training
```
It is advisable to view the contents of the .json file to note the correspondence between the labels and the classes. This annotation will be useful in the inference phase of the trained StyleGAN3. Ex: AKIEC corresponds to label 0, BCC corresponds to label 1 etc...

Now that the dataset is fully defined, let's move on to the training phase of [StyleGAN3](https://github.com/NVlabs/stylegan3.git).

```
git clone https://github.com/NVlabs/stylegan3.git
cd stylegan3
conda env create -f environment.yml
conda activate stylegan3
```
Let's run the `dataset_tool.py`, which allows you to make all the images and all the labels in a format suitable for what StyleGAN 3 expects. Adjust the desired resolution. In our case 1024x1024 pixels was used.
```
python dataset_tool.py --source /.../Dataset_Training --dest /.../Output_dataset_tool --resolution 1024x1024
```
In the path `/.../Output_dataset_tool` the final dataset will be saved to train the StyleGAN3.

Now let's continue with the training of StyleGAN3 with `train.py`. The following block is used to start a new training from scratch.
For more information about the training parameters consult the source:[StyleGAN3](https://github.com/NVlabs/stylegan3.git).
```
!python train.py --outdir /.../Output_train_StyleGAN3 \
--data /.../Output_dataset_tool \
--cfg=stylegan3-t --gpus=1 --batch=32 --gamma=32 --batch-gpu=4 --snap=5 --cond True --mirror=1
```
To resume a previous training, run the following block.
```
!python train.py --outdir /.../Output_train_StyleGAN3 \
--data /.../Output_dataset_tool \
--cfg=stylegan3-t --gpus=1 --batch=32 --gamma=32 --batch-gpu=4 --snap=5 --cond True --mirror=1 \
--resume=/.../network-snapshot-000060.pkl
```
Once you finish exercising, it may be helpful to view the FID throughout your entire workout. To do this, the `FID_visualizer.py` function is provided.

So let's enter the path where all the training outputs are located, the path where we want the FID graph to be saved, and also the total number of images in the dataset. Ex: 25202.
```
cd Fancy_Augmentation_and_Classification
python FID_visualizer.py --path_results /.../Output_train_StyleGAN3 \
--path_output /.../FID_graph \
--dataset_images_num 25202
```
An example FID graph of a complete workout from scratch is shown below.

![This is an image](https://myoctocat.com/assets/images/base-octocat.svg)




