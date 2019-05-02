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
import skimage.measure
import matplotlib.patches as patches
from scipy.spatial import distance_matrix
from skimage.draw import circle
import cv2
import imutils
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

from imports.utils.enums import DATA_BASE_PATH, SHAPE
from imports.utils.log_progress import log_progress
import imgaug as ia
from imgaug import augmenters as iaa

# https://stackoverflow.com/questions/9056957/correct-way-to-define-class-variables-in-python

class Visualize():
    def __init__(self,df,model,pred_layer=1,input_shape=(512,768,3),masktype='hand'):
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
        self.seq_norm = iaa.Sequential([
            iaa.CLAHE(),
            iaa.LinearContrast(alpha=1.0)
        ])

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

    def get_roots(self,index):
        if index == 'random':
            raise ValueError('Random not supported here!')
        self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        return [tuple(r) for r in self.selected_row["roots"].values[0]]
            
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
        if self.mode == "image_prediction_roots":
            self.predict()
            root_coords = Evaluate.get_root_pred_coord_v1(self,self.prediction)
            if root_coords is not None:
                for c in root_coords:
                    circ = patches.Circle(c,30,facecolor='red',alpha=0.5)
                    circ2 = patches.Circle(c,4,facecolor='black')
                    ax.add_patch(circ)
                    ax.add_patch(circ2)
                if "roots" in self.selected_row:
                    for root_y in self.selected_row["roots"].values[0]/2:
                    # print(root_y)
                        circ = patches.Circle(tuple(root_y),5,facecolor='yellow')
                        ax.add_patch(circ)
            ax.imshow(self.img,cmap='gray')
            if "roots" in self.selected_row:
                ax.imshow(self.msk, alpha=0.4)
        if self.mode == 'image_prediction_contour':
            self.predict()
            ax.imshow(self.img,cmap='gray')
            CS = ax.contour(self.msk,[-1,1],colors='cyan',linewidths=3)
            if self.prediction_threshold == None:
                ax.imshow(self.prediction, alpha=0.4)
            else:
                ax.imshow(self.prediction>self.prediction_threshold, alpha=0.4)
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

        if 'mask_path' in self.df:
            if self.pred_layer == 1:
                msk = imread(self.selected_row.mask_path.values[0]+self.selected_row.name.values[0])
            elif self.pred_layer == 2:
                msk = imread('../data/00_all/masks_matlab2/'+self.selected_row['name'].values[0])
                msk = imread(self.selected_row.mask_path.values[0]+self.selected_row.name.values[0])
            
            self.msk = resize(msk,self.input_shape[:2]).reshape(*self.input_shape[:2])
            self.msk = img_as_float(self.msk)
        else:
            self.msk = np.zeros(self.input_shape[:2])
        
    def predict(self):
        if self.model.layers[0].input_shape[1:] != self.input_shape:
            raise ValueError('Modelinput and Image Shape doesnt match \n ' + 'Modelinput Shape is: ' + str(self.model.layers[0].input_shape[1:]) + '\n' + 'Defined Input Shape is: ' + str(self.input_shape))
        #self.img = exposure.equalize_adapthist(self.img, clip_limit=0.03)
        
        self.img = (self.img*255).astype("uint8")
        self.img = self.seq_norm.augment_image(self.img)
        self.img = self.img.astype(float)/255.0
        
        tmp_img = self.img.reshape(1,*self.input_shape)

        assert self.pred_layer == 1 or self.pred_layer == 2, "pred_layer number not allowed"
        
        if self.pred_layer == 1:
            self.prediction = self.model.predict(tmp_img)[:,:,:,0]
        elif self.pred_layer == 2:
            self.prediction = self.model.predict(tmp_img)[:,:,:,1]

        self.prediction = self.prediction.reshape(*self.input_shape[:2])
        if self.prediction_threshold == None:
            self.prediction = img_as_float(self.prediction)
        else:
            self.prediction = img_as_float(self.prediction>self.prediction_threshold)
            
        if self.msk is not None:
            smooth = 0.0
            y_true_f = np.ndarray.flatten(self.msk.astype(float))
            y_pred_f = np.ndarray.flatten(self.prediction.astype(float))
            intersection = np.sum(y_true_f * y_pred_f)
            self.dice_score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    # https://www.kaggle.com/gauss256/preprocess-images

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
        self.seq_norm = iaa.Sequential([
            iaa.CLAHE(),
            iaa.LinearContrast(alpha=1.0)
        ])

    def get_dice_coeff_score(self,mode='simple'):
        assert mode=='simple' or mode=='raw', 'Mode must be "simple" or "raw"'
        dice_coeffs = []
        prediction_times = [] # Just for stats
        for i in log_progress(range(len(self.df)),name='Samples to Test'):
            self.selected_row = self.df.iloc[[i]]
            self.load_data()
            t = time.time()
            self.predict()
            prediction_times.append(time.time() - t)
            dice_coeffs.append(self.__dice_score(self.prediction.reshape(1,*self.input_shape[:2],1),self.msk.reshape(1,*self.input_shape[:2],1)))

        print("Average prediction time: %.2f s" % (sum(prediction_times)/len(prediction_times)))
        if mode=='simple':
            return min(dice_coeffs), max(dice_coeffs), sum(dice_coeffs)/len(dice_coeffs)
        elif mode == 'raw':
            return dice_coeffs

    def get_iou_score(self,mode='simple'):
        iou_scores = []
        for i in log_progress(range(len(self.df)),name='Samples to Test'):
            self.selected_row = self.df.iloc[[i]]
            self.load_data()
            self.predict()
            iou_scores.append(self.__iou_score(self.prediction.reshape(1,*self.input_shape[:2],1),self.msk.reshape(1,*self.input_shape[:2],1)))
        if mode=='simple':
            return min(iou_scores), max(iou_scores), sum(iou_scores)/len(iou_scores)
        elif mode == 'raw':
            return iou_scores

    def __dice_score(self,y_true,y_pred,smooth=1e-12):
        y_true_f = np.ndarray.flatten(y_true)
        y_pred_f = np.ndarray.flatten(y_pred)
        intersection = np.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
        return score #Scalar
    
    def __iou_score(self,y_true, y_pred, smooth=1e-12):
        #https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/metrics.py
        axes = (1, 2)

        intersection = np.sum(y_true * y_pred, axis=axes)
        union = np.sum(y_true + y_pred, axis=axes) - intersection
        iou = (intersection + smooth) / (union + smooth)

        # mean per image
        iou = np.mean(iou, axis=0)

        return iou

    def get_root_pred_coord_v1(self,prediction,threshold=0.8):
        prediction = prediction > threshold
        labels = skimage.measure.label(prediction)
        roots_pred = skimage.measure.regionprops(labels)
        roots_pred = [r for r in roots_pred if r.area > 1500]
        roots_pred = [r.centroid for r in roots_pred]
        roots_pred = [list(p) for p in roots_pred] #Convert to same format
        roots_pred = [p[::-1] for p in roots_pred] #Flipp X,Y
        return roots_pred

    def get_root_pred_coord_v2(self,prediction):
        # https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
        pred = (prediction*255).astype("uint8")
        tmp = np.zeros((512,768,3),dtype="uint8")
        tmp[:,:,0] = pred
        tmp[:,:,1] = pred
        tmp[:,:,2] = pred
        pred = tmp

        # load the image and perform pyramid mean shift filtering
        # to aid the thresholding step
        shifted = cv2.pyrMeanShiftFiltering(pred, 21, 51)
        
        # convert the mean shift image to grayscale, then apply
        # Otsu's thresholding
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
        
        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)
        #print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

        roots_pred = []

        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background'
            # so simply ignore it
            if label == 0:
                continue
            # otherwise, allocate memory for the label region and draw
            # it on the mask
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            if np.sum(mask[labels == label]) < 300000:
                continue
            
            #print(np.sum(mask[labels == label]))
        
            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
        
            # draw a circle enclosing the object
            ((x, y), r) = cv2.minEnclosingCircle(c)

            roots_pred.append((x,y))

        roots_pred = [list(p) for p in roots_pred]
        return roots_pred
        
    def get_root_precicion_v2(self,index,tolerance=60,print_distance_matrix=False):
        assert self.pred_layer == 2, "Wrong Prediction Layer"
        self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        self.load_data()
        self.predict()
        roots_true = (list(self.selected_row["roots"].values/2)[0])
        roots_true = [list(list(t)) for t in roots_true] #Convert to same format
        roots_pred = self.get_root_pred_coord_v1(self.prediction)
        
        if len(roots_pred) > 0 and len(roots_true) > 0:
            # if there are both roots in the image and we are also predict at least one:
            ds = distance_matrix(roots_true, roots_pred)
            # axis0 = number ground truth
            # axis1 = number predictions
            if print_distance_matrix == True:
                print(ds)

            tP = sum(ds.min(axis=0)<=tolerance)
            fN = sum(ds.min(axis=1)>tolerance)
            fP = ds.shape[1]-tP

            precision = tP / (tP+fP)
            recall = tP / (tP+fN)

        elif len(roots_true) == 0 and len(roots_pred) > 0:
            # If there are no roots in image but we predict some:
            tP = 0
            fN = 0
            fP = len(roots_pred)
            precision = 0.0
            recall = 0.0

        elif len(roots_true) > 0 and len(roots_pred) == 0:
            # If there are roots in the image but we predict none:
            tP = 0
            fN = len(roots_true)
            fP = 0
            precision = 0.0
            recall = 0.0

        else:
            # If there are neither roots in the image and prediction:
            tP = 0
            fN = 0
            fP = 0
            precision = 1.0
            recall = 1.0
        
        return tP, fP, fN, precision, recall

    def get_root_precicion(self,index,tolerance=60,print_distance_matrix=False):
        assert self.pred_layer == 2, "Wrong Prediction Layer"
        self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        self.load_data()
        self.predict()
        roots_true = (list(self.selected_row["roots"].values/2)[0])
        roots_true = [list(list(t)) for t in roots_true] #Convert to same format
        roots_pred = self.get_root_pred_coord_v2(self.prediction)

        if len(roots_pred) > 0 and len(roots_true) > 0:
            # if there are both roots in the image and we are also predict at least one:
            ds = distance_matrix(roots_true, roots_pred)
            # axis0 = number ground truth
            # axis1 = number predictions
            if print_distance_matrix == True:
                print(ds)

            all_errors = sum(ds.min(axis=1)>tolerance)
            tP = sum(ds.min(axis=1)<=tolerance)
            combined = 0
            #print(tP)
            if tP > ds.shape[1]:
                combined = abs(tP-ds.shape[1])
                tP -= combined

            if all_errors > 0:
                fP = ds.shape[1]-tP
                fN = ds.shape[0]-tP
                fN -= combined
            else:
                fP = 0
                fN = 0

            precision = tP / (tP+fP)
            recall = tP / (tP+fN)

        elif len(roots_true) == 0 and len(roots_pred) > 0:
            # If there are no roots in image but we predict some:
            tP = 0
            fN = 0
            fP = len(roots_pred)
            precision = 0.0
            recall = 0.0

        elif len(roots_true) > 0 and len(roots_pred) == 0:
            # If there are roots in the image but we predict none:
            tP = 0
            fN = len(roots_pred)
            fP = 0
            precision = 0.0
            recall = 0.0

        else:
            # If there are neither roots in the image and prediction:
            tP = 0
            fN = 0
            fP = 0
            precision = 1.0
            recall = 1.0
        
        return tP, fP, fN, precision, recall
        
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
