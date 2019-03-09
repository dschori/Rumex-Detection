import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure, imread, imsave, imshow
from skimage.transform import rescale, resize
from skimage import img_as_float

from imports.utils.enums import DATA_BASE_PATH, SHAPE
from imports.utils.log_progress import log_progress

# https://stackoverflow.com/questions/9056957/correct-way-to-define-class-variables-in-python

class Visualize():
    def __init__(self,df,data_set,model):
        self.df = df
        self.data_set = data_set
        self.model = model
        self.figsize = (10,10)
        self.prediction_threshold = 0.95
        self.index = None
        self.df_row_at_index = None
        self.mode = None
        self.img = None
        self.msk = None
        self.prediction = None
        self.dice_score = None

    def get_data_set(self):
        return self.data_set

    def get_image(self,index):
        if index == 'random':
            raise ValueError('Random not supported here!')
        self.index = index
        self.__load_data()
        return self.img

    def get_mask(self,index):
        if index == 'random':
            raise ValueError('Random not supported here!')
        self.index = index
        self.__load_data()
        return self.msk

    def get_prediction(self,index):
        if index == 'random':
            raise ValueError('Random not supported here!')
        self.index = index
        self.__load_data()
        self.__predict()
        return self.prediction

    def get_false_negative_mask(self,index):
        if index == 'random':
            raise ValueError('Random not supported here!')
        self.index = index
        self.__load_data()
        self.__predict()
        false_negative_error = (self.prediction-self.msk)>0
        return img_as_float(false_negative_error)

    def get_false_positive_mask(self,index):
        if index == 'random':
            raise ValueError('Random not supported here!')
        self.index = index
        self.__load_data()
        self.__predict()
        false_positive_error = (self.prediction-self.msk)<0
        return img_as_float(false_positive_error)

    def get_full_error_mask(self,index):
        if index == 'random':
            raise ValueError('Random not supported here!')
        self.index = index
        self.__load_data()
        self.__predict()
        error = np.abs((self.prediction-self.msk))
        return error
            
    def show_single(self,index,mode):
        self.mode = mode
        _, ax = plt.subplots(1,1,figsize=self.figsize)
        self.__add_image(index,ax)

    def show_matrix(self,index,mode,rows=4):
        self.mode = mode
        
        if index == 'random':
            index = []
            n = rows*2
            for i in range(n):
                tmp_index = np.random.random_integers(0,len(self.df))
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
        plt.subplots_adjust(wspace=0.01, hspace=0)
                         
    def __make_image_row(self,index,ax):
        self.__add_image(index[0],ax[0])
        self.__add_image(index[1],ax[1])
            
    def __add_image(self,index,ax=None):
        '''
        Adds image to 'ax' Object
        '''
        
        if index == 'random':
            self.index = np.random.random_integers(0,len(self.df))
        else:
            self.index = index
            
        self.__load_data()
        
        if ax == None:
            _, ax = plt.subplots(figsize=self.figsize)
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
            self.error = np.abs((self.prediction-self.msk))
            ax.imshow(self.img, interpolation='none')
            ax.imshow(self.prediction, interpolation='none',alpha=0.2)
            ax.imshow(self.error,cmap='Reds', alpha=0.4, interpolation='none')
        if self.dice_score == None:
            self.dice_score = 0.0
        ax.set_title('Image: ' + str(self.df_row_at_index.name.values[0]) + "  Dice Coeff: " + str(round(self.dice_score,2)), fontsize=15)
        ax.axis('off')
        self.dice_score = None
        
    def __load_data(self):

        self.df_row_at_index = self.df[self.df['name'].str.contains(str(self.index))]
        
        if len(self.df_row_at_index) == 0:
            raise ValueError('Image not found')
        elif len(self.df_row_at_index) > 1:
            raise ValueError('Index not unique')
        
        img= imread(self.df_row_at_index.image_path.values[0]+self.df_row_at_index.name.values[0])
        msk = imread(self.df_row_at_index.mask_path.values[0]+self.df_row_at_index.name.values[0])
        
        self.img = resize(img,(SHAPE[0],SHAPE[1])).reshape(*SHAPE,3)
        self.img = img_as_float(self.img)
        self.msk = resize(msk,(SHAPE[0],SHAPE[1])).reshape(*SHAPE)
        self.msk = img_as_float(self.msk)

        self.index_loaded = self.index
        
    def __predict(self):
        tmp_img = self.img.reshape(1,*SHAPE,3)
        self.prediction = self.model.predict(tmp_img)
        self.prediction = self.prediction.reshape(*SHAPE)
        if self.prediction_threshold == None:
            self.prediction = img_as_float(self.prediction)
        else:
            self.prediction = img_as_float(self.prediction>self.prediction_threshold)
        smooth = 1.0
        y_true_f = np.ndarray.flatten(self.msk.astype(float))
        y_pred_f = np.ndarray.flatten(self.prediction.astype(float))
        intersection = np.sum(y_true_f * y_pred_f)
        self.dice_score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

class Evaluate(Visualize):
    '''
    Evaluate specific Model
    Target: tbd
    '''
    def __init__(self,df,model):
        self.df = df
        self.img = None

    def __find_roots(self):
        pass

    def __eval_circle_model(self):
        for _, row in self.df.iterrows():
            self.img = imread(row['image_path']+row['name'])
            self.__predict()
            self.__find_roots()
            roots = row['roots']
            for root in roots:
                pass
