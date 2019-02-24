import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure, imread, imsave, imshow
from skimage.transform import rescale, resize

from imports.utils.enums import DATA_BASE_PATH, SHAPE
from imports.utils.log_progress import log_progress


class Visualize():
    def __init__(self,data_set,model):
        self.data_set = data_set
        self.model = model
        self.figsize = (10,10)
        self.prediction_threshold = None
        self.index = None
        self.mode = None
        self.img = None
        self.msk = None
        self.prediction = None
        self.indices = None
        self.directory = self.__get_directory()
        self.file_list = os.listdir(self.directory+'Images/Placken')
            
    def show_single(self,index,mode):
        self.mode = mode
        _, ax = plt.subplots(1,1,figsize=self.figsize)
        self.__add_image(index,ax)
        return self.img, self.prediction

    def show_matrix(self,index,mode,rows=4):
        self.mode = mode
        
        if index == 'random':
            index = []
            n = rows*2
            for i in range(n):
                tmp_index = np.random.random_integers(0,len(self.file_list))
                tmp_index = self.file_list[tmp_index]
                index.append(tmp_index)
        else:
            n = len(index)
        #Add None if odd:
        if n%2 != 0:
            index.append(None)
                         
        _, ax = plt.subplots(int(n/2),2,figsize=(15,3*n))
        
        for i in log_progress(range(len(ax)),every=1,name='Rows'):
            ind = index[2*i:2*i+2]
            self.__make_image_row(ind,ax[i])  
            
    def __add_image(self,index,ax=None):
        '''
        Adds image to 'ax' Object
        '''
        self.file_list = os.listdir(self.directory+'Images/Placken')
        
        if index == 'random':
            self.index = np.random.random_integers(0,len(self.file_list))
            self.index = self.file_list[self.index]
        else:
            self.index = index
            
        self.__load_data()
        
        if ax == None:
            fig, ax = plt.subplots(figsize=self.figsize)
        if self.mode == "image":
            ax.imshow(self.img)
        if self.mode == "mask":
            ax.imshow(self.msk)
        if self.mode == "image_mask":
            ax.imshow(self.img)
            ax.imshow(self.msk,cmap="terrain",alpha=0.4)
        if self.mode == "image_prediction":
            self.__predict()
            ax.imshow(self.img)
            if self.prediction_threshold == None:
                ax.imshow(self.prediction, alpha=0.4)
            else:
                ax.imshow(self.prediction>self.prediction_threshold, alpha=0.4)
        if self.mode == "image_prediction_error":
            self.__predict()
            error = np.equal(self.prediction>0.95,self.msk<1)
            ax.imshow(self.img)
            if self.prediction_threshold == None:
                ax.imshow(self.prediction, alpha=0.4)
            else:
                ax.imshow(self.prediction>self.prediction_threshold, alpha=0.4)
            ax.imshow(error,cmap='Reds', alpha=0.3)
            overlap = np.invert(error)
            dice = ((overlap.sum()/(error.shape[0]*error.shape[1])))
            dice = overlap.sum()/(self.msk.sum())
            if dice > 1:
                dice = 1/dice
        ax.set_title('Image Nr: ' + str(self.file_list[self.indices[0]]))
        
                         
    def __make_image_row(self,index,ax):
        self.__add_image(index[0],ax[0])
        self.__add_image(index[1],ax[1])
        
    def __get_directory(self):
        if self.data_set == "train":
            directory = DATA_BASE_PATH+"/01_Train/"
        elif self.data_set == "val":
            directory = DATA_BASE_PATH+"/02_Val/"
        elif self.data_set == "test":
            directory = DATA_BASE_PATH+"/03_Test/"
        elif self.data_set == "augmented":
            directory = DATA_BASE_PATH+"/Augmented/"
        else:
            raise ValueError('Invalid "data_set"')
        return directory
        
    def __load_data(self):
        self.indices = [i for i, s in enumerate(self.file_list) if str(self.index) in s]
        
        if len(self.indices) == 0 or len(self.indices) > 1:
            raise ValueError('Image not found')
        
        img= imread(self.directory+"Images/Placken/"+self.file_list[self.indices[0]])
        msk = imread(self.directory+"Masks/Placken/"+self.file_list[self.indices[0]])
        
        self.img = resize(img,(SHAPE[0],SHAPE[1])).reshape(*SHAPE,3)
        self.msk = resize(msk,(SHAPE[0],SHAPE[1])).reshape(*SHAPE)
        
    def __predict(self):
        tmp_img = self.img.reshape(1,*SHAPE,3)
        self.prediction = self.model.predict(tmp_img)
        self.prediction = self.prediction.reshape(*SHAPE)
