# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:05:48 2022

@author: Antonio Vispi
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

def FID_graph(path_results,path_output,dataset_images_num):

    os.makedirs(path_output, exist_ok = True)

    all_FID = []
    
    directory = os.listdir(path_results)
    for i in range(0,len(directory)):
      with open(path_results+'/'+directory[i]+'/log.txt', 'r') as file:
        data = file.read().rstrip()
      indices = [index for index in range(len(data)) if data.startswith('fid50k_full',index)]
    
      for j in range(0,len(indices),2):
        support = data[indices[j]:indices[j+1]]
        all_FID.append(support[14:20])
    for i in range(0,len(all_FID)):
      all_FID[i]=float(all_FID[i])

    Epoch_Training_orig = np.linspace(0, len(all_FID*20000)/(float(dataset_images_num)), num=len(all_FID))

    for i in range (0,len(all_FID)):
      if all_FID[i] == min(all_FID):
        min_epoch = i*20000/(float(dataset_images_num))
        position = i

    print('Il minor FID riscontrato è pari a '+str(min(all_FID))+', e si è verificato all epoca numero '+str(round(min_epoch))+', che corrisponde a '+str(position*20)+' kimg.' )
    
    figure(figsize=(15,10))
    plt.yticks(np.arange(0, max(all_FID), round(max(all_FID)/30,2)))
    plt.plot(Epoch_Training_orig,all_FID, color ='r',label = 'FID over epochs', marker='-gx', ms = 10, markevery=[position])
    
    plt.xlabel('Epochs')
    plt.ylabel('FID')
    plt.title('FID during training')
    plt.rcParams.update({'font.size': 16})
    plt.legend()
    
        
    plt.savefig(path_output+'/Graph.png')
    print('\n')
    print('The graph has been saved in the destination path.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_results', help='output path containing all training sessions')
    parser.add_argument('--path_output', help='path for saving the FID graph as a .png file')
    parser.add_argument('--dataset_images_num', help='Number of images in the dataset.')


    args = parser.parse_args()
    FID_graph(args.path_results,args.path_output,args.dataset_images_num)

if __name__ == '__main__':
    main()
