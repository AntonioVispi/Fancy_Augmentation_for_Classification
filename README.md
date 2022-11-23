# Fancy_Augmentation_for_Classification
Tool for creating a balanced dataset, augmenting it in the classic way, and then using it to train StyleGAN3. Once the StyleGAN3 has been trained, it will be used to generate synthetic images, which will allow to increase the images available to train an MMClassification classifier (EfficientNET-B4).

Before discussing the steps, let's say that the repository is divided into two parts: the first dedicated to the reproduction of the result obtained with StyleGAN3, the second dedicated to the reproduction of the result obtained with MMClassification.

This work was done specifically for dermoscopic images. Images are from the ISIC dataset.

# Getting started with GAN section

The first phase is the creation of the dataset. please install:
```
git clone https://github.com/AntonioVispi/Fancy_Augmentation_for_Classification.git
cd Fancy_Augmentation_for_Classification
pip install -U albumentations
```
(Optional step) If among the classes of your dataset, there is one class that is too numerous compared to the others, it is possible to select a subset of better images. The number of images you want will be selected. The selected images have the best resolution of all.

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
## Note: It is advisable to view the contents of the .json file obtained in this last step to note the correspondence between the labels and the classes. This annotation will be useful in the inference phase of the trained StyleGAN3. Ex: AKIEC corresponds to label 0, BCC corresponds to label 1 etc...
After this last operation, the folder with the dataset should look like this:
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
This function will display the lowest FID value and the corresponding epoch on the screen.

An example FID graph of a complete training from scratch is shown below.
![FID_Graph](https://user-images.githubusercontent.com/102518682/203628618-8aa4ab53-136b-423c-96b4-c3354203f0a5.jpg)

At this point it is possible to make the inference of the trained model as shown in the example below. 

The labels of the specific case are inside the .json file of the previous section expressed in bold.
You have to insert the label corresponding to the desired class in the entry: `class`.
`trunc` stands for truncation, by default it is set to 1. For more information consult [StyleGAN3](https://github.com/NVlabs/stylegan3.git). 

Note: the following labels are just an example
```
#   AKIEC = 0   #
#   KL = 1      #
#   MEL = 2     #
#   NV = 3      #
#   BCC = 4     #

cd stylegan3
python gen_images.py --outdir=/.../NV_fake \
--trunc=1 --seeds='1-5' \                                 # generation of 5 images that will be saved in the folder /.../NV_fake
--network=/.../network-snapshot-000080.pkl --class=3      # in this example it corresponds to class NV
```
Below is shown the example of some photos belonging to the melanoma class (MEL from ISIC), generated by the StyleGAN3 trained according to the procedure expressed up to this point.

![Norm_fake_Mel](https://user-images.githubusercontent.com/102518682/203645252-4ecc3917-1684-4b5d-9ae1-faa92daf527f.jpg)


