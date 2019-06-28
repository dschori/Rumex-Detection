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

from utils.utils import *
from utils.log_progress import *

import imgaug as ia
from imgaug import augmenters as iaa

# https://stackoverflow.com/questions/9056957/correct-way-to-define-class-variables-in-python

class Visualize():
    def __init__(self,df,model,predictiontype='leaf',input_shape=(512,768,3),masktype='leaf'):
        self.df = df
        self.model = model
        self.predictiontype = predictiontype
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
        assert masktype=='leaf' or masktype=='root', 'Masktype not allowed'
        self.masktype = masktype
        self.seq_norm = iaa.Sequential([
            iaa.CLAHE(),
            iaa.LinearContrast(alpha=1.0)
        ])

    def get_image(self,index="random"):
        """
        Get Image

        :param int or str "index": index of Image to load
        :return: image
        
        """
        if index == 'random':
            self.selected_row = self.df.sample(1)
        else:
            self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        self.load_data()
        self.__process_input()
        return self.img

    def get_mask(self,index):
        """
        Get Mask

        :param int or str "index": index of Mask to load
        :return: image
        
        """
        if index == 'random':
            self.selected_row = self.df.sample(1)
        else:
            self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        self.load_data()
        return self.msk

    def get_roots(self,index):
        """
        Get Roots Coordinates

        :param int or str "index": index of Image to load roots from
        :return: Root Coordinates as List of Tuples (x,y)
        """
        if index == 'random':
            self.selected_row = self.df.sample(1)
        else:
            self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        return [tuple(r) for r in self.selected_row["roots"].values[0]]
            
    def get_prediction(self,index):
        """
        Get Prediction of Image

        :param int or str "index": index of Image to make prediction
        :return: Prediction as Image
        """
        if index == 'random':
            self.selected_row = self.df.sample(1)
        else:
            self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        self.load_data()
        self.predict()
        return self.prediction
            
    def show_single(self,index,mode):
        """
        Show Single Image

        :param int or str "index": index of Image to show
        :param str "mode": 
                image : shows only image
                mask : shows only mask
                image_mask : shows image with overlayed mask
                image_prediction : shows image with overlayed prediction
                image_prediction_roots : shows image with GT mask and predicted roots
                image_prediction_contour : shows image with predicted segmentation and GT contours
        :return: No return Value
        """
        self.mode = mode
        if index == 'random':
            self.selected_row = self.df.sample(1)
        else:
            self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        _, ax = plt.subplots(1,1,figsize=self.figsize)
        self.__add_image(self.selected_row,ax)

    def show_matrix(self,index,mode,rows=4):
        """
        Show a rows x 2 Matrix of images

        :param List of int or str: List of indexes to show, or "random"
        :param str "mode": 
                image : shows only image
                mask : shows only mask
                image_mask : shows image with overlayed mask
                image_prediction : shows image with overlayed prediction
                image_prediction_roots : shows image with GT mask and predicted roots
                image_prediction_contour : shows image with predicted segmentation and GT contours
        :param int "row": how much rows should be displayd
        :return: No return Value
        """
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
                ax.imshow(self.prediction>0.7, alpha=0.5)
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
            if self.masktype == 'leaf':
                #msk = imread(self.selected_row.mask_path.values[0]+self.selected_row.name.values[0])
                msk = imread('../data/00_all/masks_leaf-segmentation/'+self.selected_row['name'].values[0])
            elif self.masktype == 'root':
                msk = imread('../data/00_all/masks_root-estimation/'+self.selected_row['name'].values[0])
                #msk = imread(self.selected_row.mask_path.values[0]+self.selected_row.name.values[0])
            
            self.msk = resize(msk,self.input_shape[:2]).reshape(*self.input_shape[:2])
            self.msk = img_as_float(self.msk)
        else:
            self.msk = np.zeros(self.input_shape[:2])

    def __process_input(self):
        self.img = (self.img*255).astype("uint8")
        self.img = self.seq_norm.augment_image(self.img)
        self.img = self.img.astype(float)/255.0
        
    def predict(self):
        if self.model.layers[0].input_shape[1:] != self.input_shape:
            raise ValueError('Modelinput and Image Shape doesnt match \n ' + 'Modelinput Shape is: ' + str(self.model.layers[0].input_shape[1:]) + '\n' + 'Defined Input Shape is: ' + str(self.input_shape))
                
        self.img = (self.img*255).astype("uint8")
        self.img = self.seq_norm.augment_image(self.img)
        self.img = self.img.astype(float)/255.0
        
        tmp_img = self.img.reshape(1,*self.input_shape)

        if self.predictiontype == 'leaf':
            self.prediction = self.model.predict(tmp_img)[:,:,:,0]
        elif self.predictiontype == 'root':
            try:
                self.prediction = self.model.predict(tmp_img)[:,:,:,1]
            except IndexError:
                self.prediction = self.model.predict(tmp_img)[:,:,:,0]

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

    def __normalize(self,img):
        img -= img.mean()
        img /= (img.std() +1e-5)
        img *= 0.1

        img += 0.5
        img = np.clip(img,0,1)

        return img

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
    def __init__(self,df,input_shape,model,predictiontype,masktype='leaf'):
        #Visualize.__init__(self,df,model,masktype='hand')
        self.df = df
        self.input_shape = input_shape
        self.model = model
        self.predictiontype = predictiontype
        self.masktype = masktype
        self.prediction_threshold = None
        self.seq_norm = iaa.Sequential([
            iaa.CLAHE(),
            iaa.LinearContrast(alpha=1.0)
        ])

    def get_seg_eval_metrics(self,prediction_threshold=0.7,dc_threshold=0.7,print_output=False):
        DCs = []
        TPs = []
        FPs = []
        FNs = []
        names = []
        for i in log_progress(range(len(self.df)),name="Samples to Test"):
            self.selected_row = self.df.iloc[[i]]
            self.load_data()
            self.predict()
            pred = self.prediction > prediction_threshold
            msk = self.msk > 0.5
            DC = self.__dice_score(msk,pred)
            pred = pred.flatten()
            msk = msk.flatten()
            TP = np.sum(pred == msk) / len(msk)
            FP = 0
            for gt,p in zip(msk,pred):
                if p == 1 and gt == 0:
                    FP += 1
            FP /= len(msk)
            FN = 0
            FN = 0 if DC > dc_threshold else 1
            #for gt,p in zip(msk,pred):
            #    if p == 0 and gt == 1:
            #        FN += 1
            #FN /= len(msk)
            name = self.df.iloc[[i]].name
            DCs.append(DC)
            TPs.append(TP)
            FPs.append(FP)
            FNs.append(FN)
            names.append(name)
            if print_output:
                print(str(DC) + " | " + str(TP) + " | " + str(FP) + " | " + str(FN) + " | " + str(name))
        return DCs, TPs, FPs, FNs, names 

    def get_dice_score(self, index, prediction_threshold=0.8):
        """
        Get dice coefficent of a prediction from a single image

        :param int "index": index of image to load
        :return: dice score
        """
        assert index != 'random', "Random not supported here!"
        self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        self.load_data()
        self.predict()
        pred = self.prediction > prediction_threshold
        msk = self.msk > 0.5
        return self.__dice_score(msk, pred)

    def get_iou_score(self, index, prediction_threshold=0.8):
        """
        Get iou score of a prediction from a single image

        :param int "index": index of image to load
        :return: iou score
        """
        assert index != 'random', "Random not supported here!"
        self.selected_row = self.df[self.df['name'].str.contains(str(index))]
        self.load_data()
        self.predict()
        pred = self.prediction > prediction_threshold
        msk = self.msk > 0.5
        return self.img, msk, pred, self.__dice_score(msk, pred)

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

    def get_iou_score_v0(self,mode='simple'):
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
        y_true = y_true.reshape(1,*self.input_shape[:2],1)
        y_pred = y_pred.reshape(1,*self.input_shape[:2],1)
        axes = (1, 2)

        intersection = np.sum(y_true * y_pred, axis=axes)
        union = np.sum(y_true + y_pred, axis=axes) - intersection
        iou = (intersection + smooth) / (union + smooth)

        # mean per image
        iou = np.mean(iou, axis=0)

        return iou

    def get_root_pred_coord_v1(self,prediction,threshold=0.4):
        assert self.predictiontype == "root", "Wrong Predictiontype"
        assert self.masktype == "root", "Wrong Masktype"
        prediction = prediction > threshold
        labels = skimage.measure.label(prediction)
        roots_pred = skimage.measure.regionprops(labels)
        roots_pred = [r for r in roots_pred if r.area > 500]
        roots_pred = [r.centroid for r in roots_pred]
        roots_pred = [list(p) for p in roots_pred] #Convert to same format
        roots_pred = [p[::-1] for p in roots_pred] #Flipp X,Y
        return roots_pred

    def get_root_pred_coord_v2(self,prediction):
        assert self.predictiontype == "root", "Wrong Predictiontype"
        assert self.masktype == "root", "Wrong Masktype"
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
        
    def get_root_precicion(self,index,tolerance=30,print_distance_matrix=False):
        assert self.predictiontype == "root", "Wrong Predictiontype"
        assert self.masktype == "root", "Wrong Masktype"
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

    def log_root_precision_values(self, tolerance=30, print_output=False):
        tPs = []
        fPs = []
        fNs = []
        precicions = []
        recalls = []
        test_log = []
        for roots_per_image in range(1,7):
            if print_output:
                print("Max roots_per_image: " + str(roots_per_image))
                print("TP\tFP\tFN\tPrecicion\tRecall\tImageName")
            for i, row in log_progress(self.df.iterrows(), every=1, size=len(self.df), name=str(roots_per_image)):
                if len(row["roots"]) <= roots_per_image:
                    #print(len(row["roots"]))
                    tP, fP, fN, precicion, recall = self.get_root_precicion(row["name"],tolerance=tolerance,print_distance_matrix=False)
                    if print_output:
                        print("{}\t{}\t{}\t{:1.2f}\t{:1.2f}\t{}".format(tP, fP, fN, precicion, recall, row['name']))
                    tPs.append(tP)
                    fPs.append(fP)
                    fNs.append(fN)
                    precicions.append(precicion)
                    recalls.append(recall)
            test_log.append((tPs,fPs,fNs,precicions,recalls))
            tPs = []
            fPs = []
            fNs = []
            precicions = []
            recalls = []
        return test_log
        
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
