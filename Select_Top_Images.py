# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:52:40 2022

@author: Antonio Vispi
"""

import argparse
from skimage.color import rgb2gray
import numpy as np
from scipy import ndimage
import os
import skimage.io as img

def estimate_sharpness_test(I):
  
  I = rgb2gray(I)

  # Get x-gradient in "sx"
  sx = ndimage.sobel(I,axis=0,mode='constant')
  # Get y-gradient in "sy"
  sy = ndimage.sobel(I,axis=1,mode='constant')
  # Get square root of sum of squares
  sobel=np.hypot(sx,sy)
  sharpness = np.sum(sobel)/(sx.shape[0]*sx.shape[1])

  return sharpness


def Top_images(path_in,path_out,top_images_num):

    os.makedirs(path_out, exist_ok = True)
    
    directory = os.fsencode(path_in)
      
    print('Start of selection of the '+top_images_num+' top images ...')
    lista=[]
    for file in os.listdir(directory):
      filename = os.fsdecode(file)
      image = img.imread(path_in+'/'+filename)
      Sharpness = estimate_sharpness_test(image)
      Num_pixels = image.shape[0]*image.shape[1]
      lista.append([filename,Num_pixels,Sharpness])
    
    top_images_num = int(float(top_images_num))
    
    # Sort that list by total number of pixels of image
    byPixels = sorted(lista, key=lambda item: item[1])
    byPixels = byPixels[(len(byPixels)-round(((len(byPixels)-top_images_num)/2))-top_images_num):len(byPixels)]
    
    # Sort that list by the sharpness score of image
    bySharpness = sorted(byPixels, key=lambda item: item[2])
    bySharpness = bySharpness[(len(bySharpness)-top_images_num):len(bySharpness)]
    
    print('Start of saving of the '+str(top_images_num)+' top images ...')
    i=0
    for i in range (0,len(bySharpness)):
      image = img.imread(path_in+'/'+bySharpness[i][0])
      img.imsave(path_out+'/'+bySharpness[i][0],image)
      i=i+1
    print(str(i)+' images have been saved in the path_out directory.')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in', help='input path of the specific class')
    parser.add_argument('--path_out', help='output path for saving the specific class')
    parser.add_argument('--top_images_num', help='Final number of images you want to select')


    args = parser.parse_args()
    Top_images(args.path_in,args.path_out,args.top_images_num)

if __name__ == '__main__':
    main()

