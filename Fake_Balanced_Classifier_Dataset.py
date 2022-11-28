# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 19:35:45 2022

@author: Antonio Vispi
"""

import argparse
import os
import random
import skimage.io as img
import numpy as np
from PIL import ImageFile


def counter(directory):
  k=0
  for file in os.listdir(directory):
    filename = os.fsdecode(file)
    k=k+1
  return k
def make_dataset(path_in,path_in_fake,path_out):

    directory = os.listdir(path_in)
    directory.sort()
    directory_fake = os.listdir(path_in_fake)
    directory_fake.sort()
    
    Num_Classes = len(directory)
    
    Attributes_Vector = []    # If the classes are written correctly, execution can continue.
    fake_images =[]
    percentages = []  
    
    os.makedirs(path_out, exist_ok = True)
    os.makedirs(path_out+'/test_set/test_set', exist_ok = True)
    os.makedirs(path_out+'/training_set/training_set', exist_ok = True)
    os.makedirs(path_out+'/val_set/val_set', exist_ok = True)
    for i in range(0,Num_Classes):
      os.makedirs(path_out+'/test_set/test_set/'+directory[i], exist_ok = True)
      os.makedirs(path_out+'/training_set/training_set/'+directory[i], exist_ok = True)
      os.makedirs(path_out+'/val_set/val_set/'+directory[i], exist_ok = True)
    
    for i in range (0,max(len(directory),len(directory_fake))):
      if directory[i] != directory_fake[i]:
        print('Error! Double check that the class names of the fake images\nand the initial images are the same, and in the same order.')
        for i in range(0,Num_Classes):
          os.rmdir(path_out+'/test_set/test_set/'+directory[i])
          os.rmdir(path_out+'/training_set/training_set/'+directory[i])
          os.rmdir(path_out+'/val_set/val_set/'+directory[i])
        os.rmdir(path_out+'/test_set/test_set')
        os.rmdir(path_out+'/training_set/training_set')
        os.rmdir(path_out+'/val_set/val_set')
        os.rmdir(path_out+'/test_set')
        os.rmdir(path_out+'/training_set')
        os.rmdir(path_out+'/val_set')
        os.rmdir(path_out)
      else:
        pass
    
####
    
    for i in range(0,Num_Classes):
      support = counter(path_in+'/'+(directory[i]))
      Attributes_Vector.append([directory[i],support])
    for i in range(0,Num_Classes):
      support = counter(path_in_fake+'/'+(directory[i]))
      fake_images.append([directory[i],support])
    # Check that the classes have been entered correctly
    
    
    support = sorted(Attributes_Vector, key=lambda item: item[1])
    Largest_Class_Num = support[len(support)-1][1]      # Greatest number of all classes.
    
    for i in range(0,Num_Classes):
      support = Largest_Class_Num-(Attributes_Vector[i][1])
      support = support/(fake_images[i][1])
      if support >= 1:
        support = 1
      else:
        pass
      percentages.append([directory[i],round(support, 3)])
    ImageFile.LOAD_TRUNCATED_IMAGES = True  
    print('Dataset creation is in progress...')
    
    for i in range (0,Num_Classes):
            Counter_test = 0 # Count of total training images saved by class
            Counter_train = 0 # Count of total training images saved by class
            Counter_val = 0 # Count of total training images saved by class
        
            images_num = counter(path_in_fake+'/'+directory[i])
        
            quotaparte_20 = round(((images_num*percentages[i][1])/100)*20);
            quotaparte_90 = round((images_num*percentages[i][1]/100)*90);
            randomized_indices = np.random.permutation(images_num)
            all_images = os.listdir(path_in_fake+'/'+directory[i])
            
            # TEST SET
            for m in range(0,quotaparte_20 +1):
              name = all_images[randomized_indices[m]]
              image = img.imread(path_in_fake+'/'+directory[i]+'/'+name)
              img.imsave(path_out+'/test_set/test_set/'+directory[i]+'/'+name,image)
              Counter_test = Counter_test + 1 
            print('A total of '+str(Counter_test)+' fake images of the '+directory[i]+' class were saved in the test set') 
    
            # TRAINING SET
            for m in range(quotaparte_20 +1,quotaparte_90 +1):
              name = all_images[randomized_indices[m]]
              image = img.imread(path_in_fake+'/'+directory[i]+'/'+name)
              img.imsave(path_out+'/training_set/training_set/'+directory[i]+'/'+name,image)
              Counter_train = Counter_train + 1 
            print('A total of '+str(Counter_train)+' fake images of the '+directory[i]+' class were saved in the training set')
    
            # VALIDATION SET
            for m in range(quotaparte_90 + 1, round(images_num*percentages[i][1])):
              name = all_images[randomized_indices[m]]
              image = img.imread(path_in_fake+'/'+directory[i]+'/'+name)
              img.imsave(path_out+'/val_set/val_set/'+directory[i]+'/'+name,image)
              Counter_val = Counter_val + 1 
            print('A total of '+str(Counter_val)+' fake images of the '+directory[i]+' class were saved in the training set')
            print('\n')
            print('A total of '+str(Counter_test+Counter_train+Counter_val)+' fake images of the '+directory[i]+' class were saved.') 
            print('\n')
    
    for i in range (0,Num_Classes):

            Counter_test = 0 # Count of total training images saved by class
            Counter_train = 0 # Count of total training images saved by class
            Counter_val = 0 # Count of total training images saved by class
        
            images_num = counter(path_in+'/'+directory[i])
        
            quotaparte_20 = round(((images_num)/100)*20);
            quotaparte_90 = round((images_num/100)*90);
            randomized_indices = np.random.permutation(images_num)
            all_images = os.listdir(path_in+'/'+directory[i])
            
            # TEST SET
            for m in range(0,quotaparte_20 +1):
              name = all_images[randomized_indices[m]]
              image = img.imread(path_in+'/'+directory[i]+'/'+name)
              img.imsave(path_out+'/test_set/test_set/'+directory[i]+'/'+name,image)
              Counter_test = Counter_test + 1
            print('A total of '+str(Counter_test)+' original images of the '+directory[i]+' class were saved in the test set') 
    
            # TRAINING SET
            for m in range(quotaparte_20 +1,quotaparte_90 +1):
              name = all_images[randomized_indices[m]]
              image = img.imread(path_in+'/'+directory[i]+'/'+name)
              img.imsave(path_out+'/training_set/training_set/'+directory[i]+'/'+name,image)
              Counter_train = Counter_train + 1 
            print('A total of '+str(Counter_train)+' original images of the '+directory[i]+' class were saved in the training set')
    
            # VALIDATION SET
            for m in range(quotaparte_90 + 1, round(images_num)):
              name = all_images[randomized_indices[m]]
              image = img.imread(path_in+'/'+directory[i]+'/'+name)
              img.imsave(path_out+'/val_set/val_set/'+directory[i]+'/'+name,image)
              Counter_val = Counter_val + 1
            print('A total of '+str(Counter_val)+' original images of the '+directory[i]+' class were saved in the training set')
            print('\n')
            print('A total of '+str(Counter_test+Counter_train+Counter_val)+' images of the '+directory[i]+' class were saved.') 
            print('\n')
        
    print('The dataset was successfully defined or updated.')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', help='input path of the directory containing all classes of the Original images')
    parser.add_argument('--path_in_fake', help='input path of the directory containing all classes of the Fake images')
    parser.add_argument('--path_out', help='output path for saving the final dataset balanced with the fake images')

    args = parser.parse_args()
    make_dataset(args.path_in,args.path_in_fake,args.path_out)
    
if __name__ == '__main__':
    main()
