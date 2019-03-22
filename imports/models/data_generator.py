import skimage
from skimage import exposure
from skimage.color import rgb2gray
from skimage.draw import circle
from matplotlib.pyplot import imshow, imread, imsave
import keras
import numpy as np 
import pandas as pd
from scipy import ndimage as ndi

from imports.utils.enums import DATA_BASE_PATH, SHAPE

import imgaug as ia
from imgaug import augmenters as iaa
from skimage.transform import rescale, resize


# Based on Keras Sequence Class
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, augment_data=False, batch_size=2, target_size=SHAPE, shuffle=True,save_images=False, input_channels=3):
        'Initialization'
        self.df = df
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
        #print("index: " + str(index))
        #print("indexes: " + str(indexes))
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
            msk_circle = imread(row['mask_cirlce_path']+row['name'])
            
            if self.input_channels == 1:
                img = rgb2gray(img)
            # Adjust Data
            img, msk, msk_circle = self.__adjust_data(img, msk, msk_circle)
            #print(msk_circle.shape)
            imgs[ind,] = resize(img,self.target_size).reshape(*self.target_size,self.input_channels)
            masks[ind,:,:,0] = resize(msk,self.target_size).reshape(*self.target_size)
            masks[ind,:,:,1] = resize(msk_circle,self.target_size).reshape(*self.target_size)
            ind += 1

        return imgs, [masks[:,:,:,0].reshape(4,512,768,1), masks[:,:,:,1].reshape(4,512,768,1)]

    def __augmented_data_generation(self,tmp_df):
        for _, row in tmp_df.iterrows():
            img = imread(row['image_path']+row['name'])
            msk = imread(row['mask_path']+row['name'])
            batch_img_aug, batch_msk_aug = self.__augment_image_batch(img,msk)
            return batch_img_aug, batch_msk_aug
            # self.__save_batches(batch_image_aug,batch_mask_aug) 

    def __augment_image_batch(self,img,msk):
        img = skimage.img_as_ubyte(img) #Image needed as int8
        imgs = [np.copy(img) for _ in range(self.batch_size)]
        img_batches = [ia.Batch(images=imgs) for _ in range(1)]

        msk = skimage.img_as_ubyte(msk) #Image needed as int8
        msks = [np.copy(msk) for _ in range(self.batch_size)]
        msk_batches = [ia.Batch(images=msks) for _ in range(1)] 

        aug = iaa.Sequential([
            #iaa.HistogramEqualization(),
            iaa.SomeOf((1, 5), [
                iaa.GaussianBlur(sigma=(0, 1.5)),
                iaa.Fliplr(0.8),
                iaa.Flipud(0.8),
                iaa.Crop(px=(0, 16)),
                iaa.Affine(rotate=(-70, 70)),
                iaa.Add((25)),
                iaa.PiecewiseAffine(scale=(0.01, 0.04)),
            ]),
        ], random_order=True)

        aug = aug.to_deterministic() #Apply same operations to image and mask
        batch_img_aug = list(aug.augment_batches(img_batches, background=False))  # background=True for multicore aug
        batch_msk_aug = list(aug.augment_batches(msk_batches, background=False))

        imgs = np.empty((self.batch_size, *self.target_size, self.input_channels))
        masks = np.empty((self.batch_size, *self.target_size, 1))

        for image in range(self.batch_size):
                tmp_img = skimage.img_as_float(batch_img_aug[0].images_aug[image])
                tmp_img = skimage.color.rgb2gray(tmp_img)
                tmp_msk = skimage.img_as_float(batch_msk_aug[0].images_aug[image])
                tmp_msk = skimage.color.rgb2gray(tmp_msk)
                tmp_msk = tmp_msk/tmp_msk.max()
                tmp_msk = tmp_msk > 0.95
                img, msk = self.__adjust_data(tmp_img, tmp_msk)
                if self.save_images == True:
                    self.__save_data(tmp_img,tmp_msk)
                imgs[image,] = resize(tmp_img,(self.target_size[0],self.target_size[1])).reshape(*self.target_size,self.input_channels)
                masks[image,] = resize(tmp_msk,(self.target_size[0],self.target_size[1])).reshape(*self.target_size,1)

        return imgs, masks
    
    def __adjust_data(self,img,mask, msk_circle):
        # Apply HistogramEqualization:
        img = exposure.equalize_hist(img)
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
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return (img,mask,msk_circle)
        
    def __save_data(self,img,mask):
        self.save_index += 1
        imsave(DATA_BASE_PATH+'/SaveDir/' + 'img_' + str(self.save_index) + '.png',img,cmap='gray')
        imsave(DATA_BASE_PATH+'/SaveDir/' + 'msk_' + str(self.save_index) + '.png',mask,cmap='gray')
		
    def __init_info(self):
        print("Found " + str(len(self.df)) + " Files")
