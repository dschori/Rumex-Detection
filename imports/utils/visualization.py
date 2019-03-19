import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure, imread, imsave, imshow
from skimage.transform import rescale, resize
from skimage.color import rgb2gray
from skimage import img_as_float
import pandas as pd
from skimage import exposure

from imports.utils.enums import DATA_BASE_PATH, SHAPE
from imports.utils.log_progress import log_progress

# https://stackoverflow.com/questions/9056957/correct-way-to-define-class-variables-in-python

class Visualize():
    def __init__(self,df,model,input_shape=(512,768,3),masktype='hand'):
        self.df = df
        self.model = model
        self.input_shape = input_shape
        self.figsize = (10,10)
        self.prediction_threshold = 0.95
        self.selected_row = None
        self.mode = None
        self.img = None
        self.msk = None
        self.prediction = None
        self.dice_score = None
        self.img_shape = None
        self.model_shape = None
        assert masktype=='auto' or masktype=='hand', 'Masktype not allowed'
        self.masktype = masktype

    def get_image(self,index):
        if index == 'random':
            raise ValueError('Random not supported here!')
        self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        self.load_data()
        return self.img

    def get_mask(self,index):
        if index == 'random':
            raise ValueError('Random not supported here!')
        self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        self.load_data()
        return self.msk

    def get_prediction(self,index):
        if index == 'random':
            raise ValueError('Random not supported here!')
        self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        self.load_data()
        self.predict()
        return self.prediction

    def get_false_negative_mask(self,index):
        if index == 'random':
            raise ValueError('Random not supported here!')
        self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        self.load_data()
        self.predict()
        false_negative_error = (self.prediction-self.msk)>0
        return img_as_float(false_negative_error)

    def get_false_positive_mask(self,index):
        if index == 'random':
            raise ValueError('Random not supported here!')
        self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        self.load_data()
        self.predict()
        false_positive_error = (self.prediction-self.msk)<0
        return img_as_float(false_positive_error)

    def get_full_error_mask(self,index):
        if index == 'random':
            raise ValueError('Random not supported here!')
        self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        self.load_data()
        self.predict()
        error = np.abs((self.prediction-self.msk))
        return error
            
    def show_single(self,index,mode):
        self.mode = mode
        if index == 'random':
            self.selected_row = self.df.sample(1)
        else:
            self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        _, ax = plt.subplots(1,1,figsize=self.figsize)
        self.__add_image(self.selected_row,ax)

    def show_matrix(self,index,mode,rows=4):
        self.mode = mode
        # Create empty header:
        selected_rows = pd.DataFrame().reindex_like(self.df).head(0)
        if index == 'random':
            n = rows*2
            selected_rows = selected_rows.append(self.df.sample(n))
        else:
            n = len(index)
            rows = int(n/2)
            if n <= 2:
                raise ValueError('Index length must be greater then 2')
            if n % 2 != 0:
                raise ValueError('Index length must be eval')
            for i in index:
                selected_rows = selected_rows.append(self.df[self.df['name'].str.contains(str(i))], ignore_index=True)
                         
        _, ax = plt.subplots(int(n/2),2,figsize=(15,3*n))
        
        for i in log_progress(range(rows),every=1,name='Rows'):
            rows = selected_rows[2*i:2*i+2]
            self.__make_image_row(rows,ax[i])
        plt.subplots_adjust(wspace=0.01, hspace=0)
                         
    def __make_image_row(self,rows,ax):
        self.__add_image(rows.iloc[[0]],ax[0])
        self.__add_image(rows.iloc[[1]],ax[1])
            
    def __add_image(self,row,ax=None):
        '''
        Adds image to 'ax' Object
        '''
        self.selected_row = row
        self.load_data()
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
            self.predict()
            ax.imshow(self.img,cmap='gray')
            if self.prediction_threshold == None:
                ax.imshow(self.prediction, alpha=0.4)
            else:
                ax.imshow(self.prediction>self.prediction_threshold, alpha=0.4)
        if self.mode == "image_prediction_error":
            self.predict()
            self.error = np.abs((self.prediction-self.msk))
            ax.imshow(self.img, interpolation='none')
            ax.imshow(self.prediction, interpolation='none',alpha=0.2)
            ax.imshow(self.error,cmap='Reds', alpha=0.4, interpolation='none')
        if self.mode == 'normalized_gray':
            norm = rgb2gray(self.img)
            norm = (norm-np.mean(norm))/np.std(norm)
            ax.imshow(norm,cmap="gray") 
        if self.dice_score == None:
            self.dice_score = 0.0
        ax.set_title('Image: ' + str(self.selected_row.name.values[0]) + "  Dice Coeff: " + str(round(self.dice_score,2)), fontsize=15)
        ax.axis('off')
        self.dice_score = None
        
    def load_data(self):
        if len(self.selected_row) == 0:
            raise ValueError('Image not found, index not in dataframe')
        img= imread(self.selected_row.image_path.values[0]+self.selected_row.name.values[0])
        if self.masktype == 'hand':
            msk = imread(self.selected_row.mask_path.values[0]+self.selected_row.name.values[0])
        elif self.masktype == 'auto':
            msk = imread(self.selected_row.mask_cirlce_path.values[0]+self.selected_row.name.values[0])[:,:,0]

        img = self.__adjust_data(img)

        if self.input_shape[2] == 1: #grayscale
            self.img = resize(img,self.input_shape[:2]).reshape(*self.input_shape[:2])
        if self.input_shape[2] == 3: #rgb
            self.img = resize(img,self.input_shape[:2]).reshape(*self.input_shape)
        self.img = img_as_float(self.img)
        self.msk = resize(msk,self.input_shape[:2]).reshape(*self.input_shape[:2])
        self.msk = img_as_float(self.msk)
        
    def predict(self):
        if self.model.layers[0].input_shape[1:] != self.input_shape:
            raise ValueError('Modelinput and Image Shape doesnt match \n ' + 'Modelinput Shape is: ' + str(self.model.layers[0].input_shape[1:]) + '\n' + 'Defined Input Shape is: ' + str(self.input_shape))

        tmp_img = self.img.reshape(1,*self.input_shape)
        self.prediction = self.model.predict(tmp_img)
        self.prediction = self.prediction.reshape(*self.input_shape[:2])
        if self.prediction_threshold == None:
            self.prediction = img_as_float(self.prediction)
        else:
            self.prediction = img_as_float(self.prediction>self.prediction_threshold)

        smooth = 0.0
        y_true_f = np.ndarray.flatten(self.msk.astype(float))
        y_pred_f = np.ndarray.flatten(self.prediction.astype(float))
        intersection = np.sum(y_true_f * y_pred_f)
        self.dice_score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    def __adjust_data(self,img):
        if self.input_shape[2] == 1: #grayscale
            img = rgb2gray(img)
            img = exposure.equalize_hist(img)
        elif self.input_shape[2] == 3: #rgb
            pass #TODO
        return img

class Evaluate(Visualize):
    '''
    Evaluate specific Model
    Target: tbd
    '''
    def __init__(self,df,model,masktype='hand'):
        Visualize.__init__(self,df,model,masktype='hand')

    def get_dice_coeff(self,mode='simple'):
        assert mode=='simple' or mode=='raw', 'Mode must be "simple" or "raw"'
        dice_coeffs = []
        prediction_times = [] # Just for stats
        for i in log_progress(range(len(self.df)),name='Samples to Test'):
            self.selected_row = self.df.iloc[[i]]
            self.load_data()
            t = time.time()
            self.predict()
            prediction_times.append(time.time() - t)
            dice_coeffs.append(self.dice_score)

        print("Average prediction time: %.2f s" % (sum(prediction_times)/len(prediction_times)))
        if mode=='simple':
            return min(dice_coeffs), max(dice_coeffs), sum(dice_coeffs)/len(dice_coeffs)
        elif mode == 'raw':
            return dice_coeffs
            
    def __find_roots(self):
        pass

    def __eval_circle_model(self):
        pass
