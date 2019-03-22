import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure, imread, imsave, imshow
from matplotlib import cm
from skimage.transform import rescale, resize
from skimage.color import rgb2gray
from skimage import img_as_float
import pandas as pd
from skimage import exposure
from PIL import Image
import skimage

from imports.utils.enums import DATA_BASE_PATH, SHAPE
from imports.utils.log_progress import log_progress

# https://stackoverflow.com/questions/9056957/correct-way-to-define-class-variables-in-python

class Visualize():
    def __init__(self,df,pred_layer,model,input_shape=(512,768,3),masktype='hand'):
        self.df = df
        self.model = model
        self.pred_layer = pred_layer
        self.input_shape = input_shape
        self.figsize = (15,15)
        self.prediction_threshold = None
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
        
        img = self.__adjust_data(img)

        if self.input_shape[2] == 1: #grayscale
            self.img = resize(img,self.input_shape[:2]).reshape(*self.input_shape[:2])
        if self.input_shape[2] == 3: #rgb
            self.img = resize(img,self.input_shape[:2]).reshape(*self.input_shape)
        self.img = img_as_float(self.img)
        self.img = self.__norm_image(self.img)

        if 'mask_path' in self.df:
            if self.masktype == 'hand':
                msk = imread(self.selected_row.mask_path.values[0]+self.selected_row.name.values[0])
            elif self.masktype == 'auto':
                msk = imread(self.selected_row.mask_cirlce_path.values[0]+self.selected_row.name.values[0])[:,:,0]
            
            self.msk = resize(msk,self.input_shape[:2]).reshape(*self.input_shape[:2])
            self.msk = img_as_float(self.msk)
        
    def predict(self):
        if self.model.layers[0].input_shape[1:] != self.input_shape:
            raise ValueError('Modelinput and Image Shape doesnt match \n ' + 'Modelinput Shape is: ' + str(self.model.layers[0].input_shape[1:]) + '\n' + 'Defined Input Shape is: ' + str(self.input_shape))

        tmp_img = self.img.reshape(1,*self.input_shape)

        assert self.pred_layer == 1 or self.pred_layer == 2, "pred_layer number not allowed"
        
        if self.pred_layer == 1:
            self.prediction = self.model.predict(tmp_img)[0]
        elif self.pred_layer == 2:
            self.prediction = self.model.predict(tmp_img)[1]

        self.prediction = self.prediction.reshape(*self.input_shape[:2])
        if self.prediction_threshold == None:
            self.prediction = img_as_float(self.prediction)
        else:
            self.prediction = img_as_float(self.prediction>self.prediction_threshold)

        #if self.msk != None:
         #   smooth = 0.0
         #   y_true_f = np.ndarray.flatten(self.msk.astype(float))
          #  y_pred_f = np.ndarray.flatten(self.prediction.astype(float))
          #  intersection = np.sum(y_true_f * y_pred_f)
          #  self.dice_score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    # https://www.kaggle.com/gauss256/preprocess-images

    def __norm_image(self,img):
        """
        Normalize PIL image

        Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
        """
        #img_y, img_b, img_r = img.convert('YCbCr').split()
        img_y = skimage.color.rgb2ycbcr(img)[:,:,0]
        img_b = skimage.color.rgb2ycbcr(img)[:,:,1]
        img_r = skimage.color.rgb2ycbcr(img)[:,:,2]

        #img_y_np = np.asarray(img_y).astype(float)
        img_y_np = img_y

        img_y_np /= 255
        img_y_np -= img_y_np.mean()
        img_y_np /= img_y_np.std()
        scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                        np.abs(np.percentile(img_y_np, 99.0))])
        img_y_np = img_y_np / scale
        img_y_np = np.clip(img_y_np, -1.0, 1.0)
        img_y_np = (img_y_np + 1.0) / 2.0
        print(img_y_np.max())

        img_y_np = (img_y_np * 255).astype(np.uint8)

        #img_y = Image.fromarray(img_y_np)
        img_y = img_y_np
        print(img_y.max())

        #img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))
        img_ybr = np.dstack((img_y, img_b, img_r))

        #img_nrm = img_ybr.convert('RGB')
        img_nrm = skimage.color.ycbcr2rgb(img_ybr)

        return img_nrm/img_nrm.max()


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
    def __init__(self,df,input_shape,model,pred_layer,masktype='hand'):
        #Visualize.__init__(self,df,model,masktype='hand')
        self.df = df
        self.input_shape = input_shape
        self.model = model
        self.pred_layer = pred_layer
        self.masktype = masktype
        self.prediction_threshold = None

    def mean_average_precicion(self,threshold=0.5):
        for i in log_progress(range(len(self.df))):
            self.selected_row = self.df.iloc[[i]]
            self.load_data()
            self.predict()
           # precicion_value = None
           # iou = self.__get_iou(y_true=self.msk,y_pred=self.prediction)
           # print(iou)
            print(self.iou_metric(y_true_in=self.msk,y_pred_in=self.prediction,print_table=True))

    def __get_iou(self,y_true,y_pred,smooth=1):
        intersection = np.sum(y_true*y_pred)
        #intersection = np.sum(np.abs(y_true * y_pred), axis=-1)
        union = (np.sum(y_true) + np.sum(y_pred)) - intersection
        iou = (intersection + smooth) / ( union + smooth)
        return iou

    
    def iou_metric(self,y_true_in, y_pred_in, print_table=False):
        # https://www.kaggle.com/aglotero/another-iou-metric

        from skimage.morphology import label
        labels = label(y_true_in > 0.5)
        y_pred = label(y_pred_in > 0.5)
        
        true_objects = len(np.unique(labels))
        pred_objects = len(np.unique(y_pred))

        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(labels, bins = true_objects)[0]
        area_pred = np.histogram(y_pred, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection

        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9

        # Compute the intersection over union
        iou = intersection / union

        # Precision helper function
        def precision_at(threshold, iou):
            matches = iou > threshold
            true_positives = np.sum(matches, axis=1) == 1   # Correct objects
            false_positives = np.sum(matches, axis=0) == 0  # Missed objects
            false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
            tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
            return tp, fp, fn

        # Loop over IoU thresholds
        prec = []
        if print_table:
            print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            if (tp + fp + fn) > 0:
                p = tp / (tp + fp + fn)
            else:
                p = 0
            if print_table:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)
        
        if print_table:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        return prec[2]


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
            
    def __find_roots(self):
        pass

    def __eval_circle_model(self):
        pass
