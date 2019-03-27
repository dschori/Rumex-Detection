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

from imports.utils.enums import DATA_BASE_PATH, SHAPE

import imgaug as ia
from imgaug import augmenters as iaa
from skimage.transform import rescale, resize


# Based on Keras Sequence Class
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df,hist_equal=False, augment_data=False, batch_size=2, target_size=SHAPE, shuffle=True,save_images=False, input_channels=3):
        'Initialization'
        self.df = df
        self.hist_equal = hist_equal
        self.augment_data = augment_data
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.save_images = save_images
        self.input_channels = input_channels
        self.save_index = 0
        self.__init_info()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        if self.augment_data == True:
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            # Find list of IDs
            tmp_df = self.df.iloc[indexes]
            imgs, masks = self.__augmented_data_generation(tmp_df)
        else:
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            # Find list of IDs
            tmp_df = self.df.iloc[indexes]
            imgs, masks = self.__data_generation(tmp_df)
        return imgs,masks

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tmp_df):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        imgs = np.empty((self.batch_size, *self.target_size, self.input_channels))
        masks = np.empty((self.batch_size, *self.target_size, 2))
        # Generate data
        ind = 0
        for _, row in tmp_df.iterrows():
            img = imread(row['image_path']+row['name'])
            msk = imread(row['mask_path']+row['name'])
            msk_circle = imread(row['row['mask_path']']+row['name'])
            
            img = resize(img,self.target_size)
            msk = resize(msk,self.target_size)
            msk_circle = resize(msk_circle,self.target_size)

            assert img.shape[2] == 3, "Number of Color Channels must be 3"
            
            if self.input_channels == 1:
                img = rgb2gray(img)
            #t = time.time()
            img, msk, msk_circle = self.__adjust_data(img, msk, msk_circle) # Adjust Data
            #elapsed = time.time() - t
            #sprint(elapsed)

            assert img.max() <= 1.0
            assert msk.max() <= 1.0
            assert msk_circle.max() <= 1.0

            imgs[ind,] = img
            masks[ind,:,:,0] = msk
            masks[ind,:,:,1] = msk_circle
            ind += 1

        return imgs, [masks[:,:,:,0].reshape(self.batch_size,*self.target_size,1), masks[:,:,:,1].reshape(self.batch_size,*self.target_size,1)]
    
    def __adjust_data(self,img,mask, msk_circle):
        if len(mask.shape) == 3:
            mask = rgb2gray(mask)
        if len(msk_circle.shape) == 3:
            msk_circle = rgb2gray(msk_circle)
        if img.max() > 1.0:
            img = img / 255
        if mask.max() > 1.0:
            mask = mask / 255
        if msk_circle.max() > 1.0:
            msk_circle = msk_circle / 255
        if self.hist_equal == True:
            img = exposure.equalize_adapthist(img, clip_limit=0.03)

        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return (img,mask,msk_circle)
        
    def __save_data(self,img,mask):
        self.save_index += 1
        imsave(DATA_BASE_PATH+'/SaveDir/' + 'img_' + str(self.save_index) + '.png',img,cmap='gray')
        imsave(DATA_BASE_PATH+'/SaveDir/' + 'msk_' + str(self.save_index) + '.png',mask,cmap='gray')
		
    def __init_info(self):
        print("Found " + str(len(self.df)) + " Files")
