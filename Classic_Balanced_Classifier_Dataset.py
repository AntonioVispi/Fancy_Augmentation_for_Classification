# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:22:07 2022

@author: Antonio Vispi
"""
import os
import argparse
import albumentations as A
import skimage.io as img
import cv2
import random
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np



def counter(directory):
  k=0
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    k=k+1
  return k

def Base_generator (path_in,path_out):
  
  os.makedirs(path_out, exist_ok = True)

  directory = os.fsencode(path_in)
  k=0
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    image = img.imread(path_in+'/'+filename)

    img.imsave(path_out+'/'+filename,image)
    k = k+1  
  print(str(k)+' images of the current class have been saved.')

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def telescope_test(image):

  img = rgb2gray(image)
  img = resize(img, (512, 512))  # make all the images square
  img = convert(img, 0, 255, np.uint8)

  mask_1 = np.ones([512,512], dtype=np.uint8)
  mask_1[15:len(mask_1)-15,15:len(mask_1)-15] = 0  #mask consisting of the outer frame 15 pixels wide
  norm_1 = np.sum(mask_1) #number of non-zero pixels of the thin mask
  immagine_1 = img * mask_1;   # I just look at the picture frame, the rest is 0

  mask_2 = np.ones([512,512], dtype=np.uint8)
  mask_2[60:len(mask_2)-60,60:len(mask_2)-60] = 0 #mask consisting of the outer frame 60 pixels wide
  norm_2 = np.sum(mask_2);   #number of non-zero pixels of the thin mask
  immagine_2 = img * mask_2;   # I just look at the picture frame, the rest is 0


  ###########  Test Section  #########
  if np.sum(immagine_2)/norm_2 <= 2.5 and np.sum(immagine_1)/norm_1 < 103:
    Test = 2;           # there is a lot of black ----> Photos with a lot of black circular border

  elif np.sum(immagine_1)/norm_1 < 103:
    Test = 0;           # there is black ----> Photo with black circular border                    

  elif np.sum(immagine_1)/norm_1 >= 103:
    Test = 1;         # little black around ----> Full Rectangular Photo
  
  return Test
  ####################################

def Labeller (path_in,path_out,class_name):  

  ##### Loop #####
    
  directory = os.fsencode(path_in)
  
  telescope = 0
  super_telescope = 0
  square = 0
  rectangle = 0
  

  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    image = img.imread(path_in+'/'+filename)

    # ----- TEST ------
 
    Test = telescope_test(image) 

    # -----------------
    if Test == 1:                  # Image without black artifact                        
      dimensions = image.shape
      dimensions = dimensions [0:2]              
      if dimensions[1] >= (dimensions[0] * 1.5):
        os.rename(path_in+'/'+filename, path_out+'/'+'R_'+class_name+'_'+filename)
        rectangle = rectangle + 1
      else:
        os.rename(path_in+'/'+filename, path_out+'/'+'Q_'+class_name+'_'+filename)
        square = square + 1 
    elif Test == 0:                # Image with black artifact
      os.rename(path_in+'/'+filename, path_out+'/'+'C_'+class_name+'_'+filename)
      telescope = telescope + 1 
    elif Test == 2:                # Image with a lot of black artifact
      os.rename(path_in+'/'+filename, path_out+'/'+'SC_'+class_name+'_'+filename)
      super_telescope = super_telescope + 1  
  return rectangle,square,telescope,super_telescope
  
  print(str(rectangle)+' images without black artifact (Rectangular images)\n'+str(telescope)+' with black artifact (Telescope images)\n'+str(super_telescope)+' with a lot of black artifact (Super-telescope images)\nwere saved.')
  print('\n')
  print('Now lets move on to the next class ...')

def fractional_augmenter(number):
    number = float(number)
    frac = number % 1
    frac = round(frac, 2)
    integ = int(number)
    x=random.randint(1, 100)
    is_true = (x < frac*100)

    if is_true:
        one_more_time = integ + 1
    else:
        one_more_time = integ
  
    return one_more_time

def auto_augmenter(r,q,c,sc,tot):
  if r+q+c+sc == tot:
    x = 1
    y = 1
    z = 1
  else: 
    z = (tot-3*r)/(2*q+2*c+sc)
    x = 2*(tot-3*r)/(2*q+2*c+sc)
    y = 2*(tot-3*r)/(2*q+2*c+sc)
  return x,y,z

def detect_group(filename):
  c = '_'
  support = [pos for pos, char in enumerate(filename) if char == c]
  group = filename[0:support[0]]
  return group

def Classic_Augmentation_Telescope (filename, path, Telescope_multiplicator):
           
     image = img.imread(path+'/'+filename)
# ---------------------------------- INITIAL TRANSFORMATION --------------------

     # Declare transformation : 1° Trasformazione : Rotate + Crop
     transform = A.Compose([
              A.Rotate (limit=[45, 45], interpolation=1, border_mode=0, value=None, mask_value=None, rotate_method='largest_box', crop_border=True, always_apply=True, p=1),
     ]) 

     transformed = transform(image=image)
     image = transformed["image"]
     #  --- ADD ---
     dimension= image.shape
     height = round((dimension[0]/100)*86)  # new height : 86%
     width = round((dimension[1]/100)*86)   # new width : 86%
     #  -----------


     Statistical_Telescope = fractional_augmenter(Telescope_multiplicator)
    
     for i in range (1, Statistical_Telescope + 1):  # Photos have increased by a factor of 'Telescope_multiplicator'
       
       if i==1:

         pass # do nothing

       else:
         # Declare transformation: 2nd Transformation. Various augmentation.
         transform = A.Compose([
                  A.PiecewiseAffine (scale=(0.025, 0.045), nb_rows=4, nb_cols=4, interpolation=5, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, always_apply=True, keypoints_threshold=0.01, p=1),
                  A.Flip(always_apply= False, p=0.5),
                  A.Transpose(always_apply= False, p=0.5),
                  A.RandomRotate90(always_apply= False, p=0.5),
                  A.CenterCrop (height, width, always_apply=True, p=1), 
         ]) 

         transformed = transform(image=image)
         transformed_image = transformed["image"]

         Pin = str(random.randint(1, 9999))+'_'
         img.imsave(path+'/'+Pin+str(i)+'_'+filename,transformed_image) # Except subsequent images submitted to Albumentation 

def Classic_Augmentation_Super_Telescope (filename, path, Super_Telescope_multiplicator):
  
     image = img.imread(path+'/'+filename)

     # ---------------------------------- INITIAL TRANSFORMATION -----------------
     # Declare an augmentation pipeline: 1st Transformation: Rotate + Crop
     transform = A.Compose([
              A.Rotate (limit=[45, 45], interpolation=1, border_mode=0, value=None, mask_value=None, rotate_method='largest_box', crop_border=True, always_apply=True, p=1),
     ])  
     # Augment an image
     transformed = transform(image=image)
     image = transformed["image"]

     dimension= image.shape
     height = round((dimension[0]/100)*72)  # Nuova altezza : 72%
     width = round((dimension[1]/100)*72)   # Nuova larghezza : 72%

     # Declare an augmentation pipeline: 2°.  Transformation : Center Crop 72 %
     transform = A.Compose([
              A.CenterCrop (height, width, always_apply=True, p=1), 
     ])  

     # Augment an image
     transformed = transform(image=image)
     image = transformed["image"]

     # ----------------------------------------------------------------------------

     Statistical_Super_Telescope = fractional_augmenter(Super_Telescope_multiplicator)
    
     for i in range (1, Statistical_Super_Telescope + 1):   # Photos have increased by a factor of 'Super_Telescope_multiplicator'
       
       if i==1:

         pass # do nothing

       else:
          # Declare an augmentation pipeline 3rd Transformation: Various augmentation. Note: The following PiecewiseAffine is more delicate than the others in this block. 
          transform = A.Compose([
                  A.PiecewiseAffine (scale=(0.025, 0.045), nb_rows=3, nb_cols=3, interpolation=5, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, always_apply=True, keypoints_threshold=0.01, p=1),
                  A.Flip(always_apply= False, p=0.5),
                  A.Transpose(always_apply= False, p=0.5),
                  A.RandomRotate90(always_apply= False, p=0.5),
          ])  

          # Augment an image
          transformed = transform(image=image)
          transformed_image = transformed["image"]

          Pin = str(random.randint(1, 9999))+'_'
          img.imsave(path+'/'+Pin+str(i)+'_'+filename,transformed_image) # Saving subsequent images subjected to Albumentation

def Classic_Augmentation_Rectangle (filename, path, Rectangle_multiplicator, current_class,Larger_Classes,Too_many_rectangles):

     image = img.imread(path+'/'+filename)

     ####### Controllo Aspect Ratio #######
     dimensions = image.shape
     dimensions = dimensions [0:2]                # I only care about height and width, not the number of channels.
     if (dimensions[1] >= (dimensions[0] * 1.5)) and (current_class not in Larger_Classes) and (current_class not in Too_many_rectangles):
       
       # --- Sliding Windows ---  
       image_L=image[0:min(dimensions),0:min(dimensions)]
       image_C=image[0:min(dimensions),round(max(dimensions)/2)-round(min(dimensions)/2):round(max(dimensions)/2)+round(min(dimensions)/2)]
       image_R=image[0:min(dimensions),max(dimensions)-min(dimensions):max(dimensions)]
       # -----------------------

       # Declare an augmentation pipeline : Albumentations vari
       transform = A.Compose([
               A.PiecewiseAffine (scale=(0.025, 0.045), nb_rows=4, nb_cols=4, interpolation=5, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, always_apply=True, keypoints_threshold=0.01, p=1),
               A.Flip(always_apply= False, p=0.5),
               A.Transpose(always_apply= False, p=0.5),
               A.RandomRotate90(always_apply= False, p=0.5),
       ])

       # --- Transformations + Saving ---
       transformed = transform(image=image_L)
       transformed_image = transformed["image"]
       label= ('RL_')
       img.imsave((path+'/'+label+filename),transformed_image)
      
       transformed = transform(image=image_C)
       transformed_image = transformed["image"]
       label= ('RC_')
       img.imsave((path+'/'+label+filename),transformed_image)

       transformed = transform(image=image_R)
       transformed_image = transformed["image"]
       label= ('RR_')
       img.imsave((path+'/'+label+filename),transformed_image)

       os.remove(path+'/'+filename)
       # -----------------------------------

     else:                                                # If, on the other hand, the image is not so rectangular ...                                              

        Statistical_Rectangle = fractional_augmenter(Rectangle_multiplicator)
        
        for i in range (1, Statistical_Rectangle + 1):                  # Photos have increased by a factor of 'Rectangular_multiplicator'

         if i==1:

           pass # do nothing

         else:
           # Declaration of an augmentation pipeline : Various Albumentations (I perform the elastic twisting before the crop, because in this way the final result is better)
           transform = A.Compose([
                     A.PiecewiseAffine (scale=(0.025, 0.045), nb_rows=4, nb_cols=4, interpolation=5, mask_interpolation=0, cval=0, cval_mask=0, mode='constant', absolute_scale=False, always_apply=True, keypoints_threshold=0.01, p=1),
                     A.Flip(always_apply= False, p=0.5),
                     A.Transpose(always_apply= False, p=0.5),
                     A.RandomRotate90(always_apply= False, p=0.5),
           ])
           transformed = transform(image=image)
           transformed_image = transformed["image"]

           new_dimensions = transformed_image.shape
           new_dimensions = new_dimensions [0:2] 
           # Declaration of an augmentation pipeline : Center Crop to make the image exactly square
           transform = A.Compose([
                     A.CenterCrop (min(new_dimensions), min(new_dimensions), always_apply=True, p=1), 
           ])  
           
           transformed = transform(image=transformed_image)
           transformed_image = transformed["image"]

           Pin = str(random.randint(1, 9999))+'_'
           label= ('Q_')
           img.imsave(path+'/'+Pin+label+str(i)+'_'+filename,transformed_image)


def Classic_Balance(path_in,path_out):
    
    os.makedirs(path_out, exist_ok = True)
    os.makedirs(path_out+'/balanced_dataset', exist_ok = True)
    os.makedirs(path_out+'/Classifier_dataset', exist_ok = True)
    
    directory = os.listdir(path_in)
    Num_Classes = len(directory)
    Attributes_Vector = []
    Larger_Classes = []
    
    for i in range(0,Num_Classes):
      support = counter(path_in+'/'+(directory[i]))
      Attributes_Vector.append([directory[i],support])
    
    support = sorted(Attributes_Vector, key=lambda item: item[1])
    Largest_Class_Num = support[len(support)-1][1]      # Greater number of all classes.
    for i in range(0,Num_Classes):
      if Attributes_Vector[i][1] >= (Largest_Class_Num/100)*95:
        Larger_Classes.append(Attributes_Vector[i][0])
    
    print('Start balancing process...')
    print('\n')
    
    # Generation of dataset copy as support
    for i in range(0,Num_Classes):
      print(directory[i]+' class is processing...')
      Base_generator(path_in+'/'+(directory[i]),path_out+'/balanced_dataset/'+(directory[i]))
     
    
    print('\n')
    print('The test is about to be performed to determine how much circular\nblack artifact the image has. The treatment of each image depends on this test.')
    
    print('The test about the black artifact over all classes is processing...')
    print('\n')
    
    details_telescopes = [] # list containing the details on the number of groups concerning the black circular artifact
    Too_many_rectangles = []
    
    for i in range(0,Num_Classes):
      [rectangle,square,telescope,super_telescope]=Labeller(path_out+'/balanced_dataset/'+(directory[i]),path_out+'/balanced_dataset/'+(directory[i]),directory[i])
      details_telescopes.append([directory[i],rectangle,square,telescope,super_telescope])
    
    for i in range(0,Num_Classes):
      if details_telescopes[i][1] >= ((Attributes_Vector[i][1]/100)*6):
        Too_many_rectangles.append(directory[i])
    
    # details_telescopes:  'Class_name', n° rectangle, n° squares, n° telescope, n° super_telescope
    
    print('\n')
    print('The test of the black artifact on the images is concluded.')
    
    print('\n')
    print('The phase of correction and augmentation of all images of all classes begins.\nThis step will take a while...')
    
    # Augmentation & corrections over all classes
    for i in range(0,Num_Classes):
          if directory[i] not in Too_many_rectangles:
            r = details_telescopes[i][1]
            q = details_telescopes[i][2]
            c = details_telescopes[i][3]
            sc = details_telescopes[i][4]
    
            print('\n')
            print(directory[i]+' class is processing...')
            [Rectangle_multiplicator,Telescope_multiplicator,Super_Telescope_multiplicator] =auto_augmenter(r,q,c,sc,Largest_Class_Num)
          else:
            r = 0
            q = details_telescopes[i][2] + details_telescopes[i][1]
            c = details_telescopes[i][3]
            sc = details_telescopes[i][4]
    
            print('\n')
            print(directory[i]+' class is processing...')
            [Rectangle_multiplicator,Telescope_multiplicator,Super_Telescope_multiplicator] =auto_augmenter(r,q,c,sc,Largest_Class_Num)
    
          for file in os.listdir(path_out+'/balanced_dataset/'+(directory[i])):
             filename = os.fsdecode(file)
             group = detect_group(filename)
    
             if group == 'R' or group == 'Q':
               Classic_Augmentation_Rectangle (filename, path_out+'/balanced_dataset/'+(directory[i]), Rectangle_multiplicator,directory[i],Larger_Classes,Too_many_rectangles)
    
             elif group == 'C':
               Classic_Augmentation_Telescope (filename, path_out+'/balanced_dataset/'+(directory[i]), Telescope_multiplicator)
    
             elif group == 'SC':
               Classic_Augmentation_Super_Telescope (filename, path_out+'/balanced_dataset/'+(directory[i]), Super_Telescope_multiplicator)  
        
    print('\n')
    for i in range(0,Num_Classes):
      print('The size of the '+directory[i]+' class is equal to '+str(counter(path_out+'/balanced_dataset/'+(directory[i]))))
    
    os.makedirs(path_out+'/Classifier_dataset/test_set/test_set', exist_ok = True)
    os.makedirs(path_out+'/Classifier_dataset/training_set/training_set', exist_ok = True)
    os.makedirs(path_out+'/Classifier_dataset/val_set/val_set', exist_ok = True)
    for i in range(0,Num_Classes):
      os.makedirs(path_out+'/Classifier_dataset/test_set/test_set/'+directory[i], exist_ok = True)
      os.makedirs(path_out+'/Classifier_dataset/training_set/training_set/'+directory[i], exist_ok = True)
      os.makedirs(path_out+'/Classifier_dataset/val_set/val_set/'+directory[i], exist_ok = True)
    
    print('Image splitting is 70% for the training set, 20% for the test set, and 10% for the validation set.')
    print('\n')
      
    # Specify the file name
    file = 'classes.txt'
    
    # Creating a file at specified location
    with open(os.path.join(path_out+'/Classifier_dataset', file), 'w') as fp:
        pass
        # To write data to new file uncomment
        string=''
        for i in range(0,Num_Classes):string = string+directory[i]+'\n'
        fp.write(string)
      
    # After creating 
    print("File .txt just created:")
    print('\n')
    print(string)
    
    
    for i in range (0,Num_Classes):
    
        Counter_test = 0 # Count of total training images saved by class
        Counter_train = 0 # Count of total training images saved by class
        Counter_val = 0 # Count of total training images saved by class
    
        images_num = counter(path_out+'/balanced_dataset/'+directory[i])
    
        quotaparte_20 = round(((images_num)/100)*20);
        quotaparte_90 = round((images_num/100)*90);
        randomized_indices = np.random.permutation(images_num)
        all_images = os.listdir(path_out+'/balanced_dataset/'+directory[i])
        
        # TEST SET
        for m in range(0,quotaparte_20 +1):
          name = all_images[randomized_indices[m]]
          image = img.imread(path_out+'/balanced_dataset/'+directory[i]+'/'+name)
          img.imsave(path_out+'/Classifier_dataset/test_set/test_set/'+directory[i]+'/'+name,image)
          Counter_test = Counter_test + 1 
        print('A total of '+str(Counter_test)+' images of the '+directory[i]+' class were saved in the test set...')
        # TRAINING SET
        for m in range(quotaparte_20 +1,quotaparte_90 +1):
          name = all_images[randomized_indices[m]]
          image = img.imread(path_out+'/balanced_dataset/'+directory[i]+'/'+name)
          img.imsave(path_out+'/Classifier_dataset/training_set/training_set/'+directory[i]+'/'+name,image)
          Counter_train = Counter_train + 1 
        print('A total of '+str(Counter_train)+' images of the '+directory[i]+' class were saved in the training set...') 
        # VALIDATION SET
        for m in range(quotaparte_90 + 1, round(images_num)):
          name = all_images[randomized_indices[m]]
          image = img.imread(path_out+'/balanced_dataset/'+directory[i]+'/'+name)
          img.imsave(path_out+'/Classifier_dataset/val_set/val_set/'+directory[i]+'/'+name,image)
          Counter_val = Counter_val + 1 
        print('A total of '+str(Counter_val)+' images of the '+directory[i]+' class were saved in the validation set...')  
        
        print('A total of '+str(Counter_test+Counter_train+Counter_val)+' images of the '+directory[i]+' class were saved.')
        print('\n')
    
    print('The dataset for MMClassification was successfully defined.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', help='input path of the directory containing all classes')
    parser.add_argument('--path_out', help='output path for saving the dataset balanced in the classic manner, and the dataset for MMClassification')

    args = parser.parse_args()
    Classic_Balance(args.path_in,args.path_out)
    
if __name__ == '__main__':
    main()