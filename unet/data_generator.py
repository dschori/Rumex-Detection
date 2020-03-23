import skimage
from skimage import exposure
from skimage.color import rgb2gray
from skimage.draw import circle
from matplotlib.pyplot import imshow, imread, imsave
import keras
import tensorflow as tf
import numpy as np 
import pandas as pd
from scipy import ndimage as ndi
import time
from skimage.util import img_as_ubyte, img_as_float
import random
import cv2

from utils.config import *


import imgaug as ia
from imgaug import augmenters as iaa
from skimage.transform import rescale, resize


# Based on Keras Sequence Class
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df,hist_equal=False, augment_data=False, batch_size=2, target_size=Config.SHAPE, shuffle=True,save_images=False, input_channels=3, output="both"):
        'Initialization'
        self.df = df
        self.hist_equal = hist_equal
        self.augment_data = augment_data
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.save_images = save_images
        self.input_channels = input_channels
        self.output = output
        self.save_index = 0
        self.mask_channels = 1
        self.__init_info()
        self.on_epoch_end()
        self.seq = iaa.Sequential([
            iaa.SomeOf((0, 4), [
                iaa.OneOf([
                    iaa.GaussianBlur(sigma=(0, 0.5)),
                    iaa.Sharpen(alpha=(0.0, 0.7)),
                ]),
                iaa.Fliplr(0.5, name="to_mask"),
                iaa.Flipud(0.5, name="to_mask"),
                #iaa.Crop(px=(0, 6), name="to_mask"),
                iaa.Affine(scale=(0.7,1.0),translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}, name="to_mask"),
            ]),
        ], random_order=True)
        self.seq_img = iaa.Sequential([
            iaa.Add((-60,60)),
        ])
        self.seq_norm = iaa.Sequential([
            iaa.CLAHE(),
            iaa.LinearContrast(alpha=1.0)
        ])

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        tmp_df = self.df.iloc[indexes]
        imgs, masks = self.__data_generation(tmp_df)
        #imgs = self.preprocess_input(imgs)
        return imgs,masks

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tmp_df):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        imgs = np.empty((self.batch_size, *self.target_size, self.input_channels),dtype=float)
        if self.output == "both":
            masks = np.empty((self.batch_size, *self.target_size, 2))
        else:
            masks = np.empty((self.batch_size, *self.target_size, 1))
        # Generate data
        ind = 0
        for _, row in tmp_df.iterrows():
            img = imread(row['image_path']+row['name'])
            
            msk_leaf = imread('../data/00_all/masks_leaf-segmentation/'+row['name'])
            msk_root = imread('../data/00_all/masks_root-estimation/'+row['name'])
            
            img = resize(img,self.target_size)
            msk_leaf = resize(msk_leaf,self.target_size)
            msk_root = resize(msk_root,self.target_size)

            #assert img.shape[2] == 3, "Number of Color Channels must be 3"
            
            #img = self.__normalize(img)

            if self.hist_equal == True:
                img = (img*255).astype("uint8")
                img = self.seq_norm.augment_image(img)
                img = img.astype(float)/255.0

            if self.augment_data == True:
                img, msk_leaf, msk_root = self.__augment_data(img,msk_leaf,msk_root)

            imgs[ind,] = img.reshape(*self.target_size, self.input_channels)

            if self.output == "both":
                masks[ind,:,:,0] = msk_leaf
                masks[ind,:,:,1] = msk_root
            if self.output == "leaf":
                masks[ind,:,:,0] = msk_leaf
            if self.output == "root":
                masks[ind,:,:,0] = msk_root
            ind += 1

        #return imgs, masks[:,:,:,0].reshape(self.batch_size,*self.target_size,1)
        return imgs, masks
    
    def __augment_data(self,img,msk_leaf,msk_root):

        img = (img*255).astype("uint8")
        msk_leaf = (msk_leaf*255).astype("uint8")
        msk_root = (msk_root*255).astype("uint8")
        
        img = self.seq_img.augment_image(img)
        
        seq_det = self.seq.to_deterministic()

        img = seq_det.augment_image(img)
        msk_leaf = seq_det.augment_image(msk_leaf.reshape(*self.target_size,1))
        msk_leaf = msk_leaf.reshape(*self.target_size)
        msk_root = seq_det.augment_image(msk_root.reshape(*self.target_size))
        msk_root = msk_root.reshape(*self.target_size)

        img = img.astype(float)/255.0
        msk_leaf = msk_leaf.astype(float)/255.0
        msk_root = msk_root.astype(float)/255.0

        return img, msk_leaf, msk_root

    def __normalize(self,img):
        img -= img.mean()
        img /= (img.std() +1e-5)
        img *= 0.1

        img += 0.5
        img = np.clip(img,0,1)

        return img
        
    def __save_data(self,img,mask):
        self.save_index += 1
        imsave(Config.DATA_BASE_PATH+'/SaveDir/' + 'img_' + str(self.save_index) + '.png',img,cmap='gray')
        imsave(Config.DATA_BASE_PATH+'/SaveDir/' + 'msk_' + str(self.save_index) + '.png',mask,cmap='gray')
		
    def __init_info(self):
        print("Found " + str(len(self.df)) + " Files")
