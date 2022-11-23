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

In this phase the .json file containing the labels is created. selected to put the same input and output path, in correspondence with the folder containing the dataset, as indicated in the previous phases.
```
git clone https://github.com/JulianPinzaru/stylegan2-ada-pytorch-multiclass-labels.git
pip install Ninja
cd stylegan2-ada-pytorch-multiclass-labels
python make_json_labels.py --input_folder=/.../Dataset_Training --output_folder=/.../Dataset_Training
```
