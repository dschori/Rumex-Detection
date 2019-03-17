import skimage
from skimage import exposure
from skimage.color import rgb2gray
from matplotlib.pyplot import imshow, imread, imsave
import keras
import numpy as np 
import pandas as pd

import imgaug as ia
from imgaug import augmenters as iaa
from skimage.transform import rescale, resize


# Based on Keras Sequence Class
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, augment_data=False, batch_size=2, target_size=SHAPE, shuffle=True, input_channels=3):
        'Initialization'
        self.df = df
        self.augment_data = augment_data
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.input_channels = input_channels
        self.__init_info()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        tmp_df = self.df.iloc[indexes]

        # train_len = 100, aug_len = 400

        if self.augment_data == True:
            imgs, masks = self.__data_generation(tmp_df)
        else:
            imgs, masks = self.__augmented_data_generation(tmp_df)

        # Generate data
        imgs, masks = self.__data_generation(tmp_df)
		
		return imgs, masks

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, tmp_df):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        imgs = np.empty((self.batch_size, *self.target_size, self.input_channels))
        masks = np.empty((self.batch_size, *self.target_size, 1))

        # Generate data
        for i, row in tmp_df.iterrows():
            img = imread(row['image_path']+row['name'])
            msk = imread(row['mask_path']+row['name'])
            img = rgb2gray(img)
            msk = rgb2gray(msk)
            print("Unique Values: " + str(np.unique(msk)))
            # Adjust Data
            img, msk = self.__adjust_data(img, msk)
            print("Unique Values: " + str(np.unique(msk)))
            imgs[i,] = resize(img,(self.target_size[0],self.target_size[1])).reshape(*self.target_size,self.input_channels)
            masks[i,] = resize(msk,(self.target_size[0],self.target_size[1])).reshape(*self.target_size,1)

        return imgs, masks
    
    def __adjust_data(self,img,mask):
        # Apply HistogramEqualization:
        img = exposure.equalize_hist(img)

        if img.max() > 1.0:
            img = img / 255
        if mask.max() > 1.0:
            mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return (img,mask)

    def __augmented_data_generation(self,tmp_df):
        for i, row in tmp_df.iterrows(): #Single loop
            img = imread(row['image_path']+row['name'])
            msk = imread(row['mask_path']+row['name'])
            batch_img_aug, batch_msk_aug = self.__augment_image_batch(img,msk)
            self.__save_batches(batch_image_aug,batch_mask_aug) 

    def __augment_image_batch(self,img,msk):
        img = skimage.img_as_ubyte(image) #Image needed as int8
        imgs = [np.copy(img) for _ in range(self.batch_size)]
        img_batches = [ia.Batch(images=imgs) for _ in range(1)]

        msk = skimage.img_as_ubyte(msk) #Image needed as int8
        msks = [np.copy(msk) for _ in range(self.batch_size)]
        msk_batches = [ia.Batch(images=msks) for _ in range(1)] 

        aug = iaa.Sequential([
            #iaa.HistogramEqualization(),
            iaa.SomeOf((1, 5), [
                iaa.GaussianBlur(sigma=(0, 1.0)),
                iaa.Fliplr(0.8),
                iaa.Flipud(0.8),
                iaa.Crop(px=(0, 16)),
                iaa.Affine(rotate=(-60, 60)),
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
                #print(im.max())
                tmp_msk = skimage.img_as_float(batch_msk_aug[0].images_aug[image])
                tmp_msk = skimage.color.rgb2gray(tmp_msk)
                tmp_msk = tmp_msk/tmp_msk.max()
                #ms = skimage.img_as_float(ms)
                #print(ms.max())
                tmp_msk = tmp_msk > 0.95
                #seed = np.copy(ms)
                #seed[1:-1, 1:-1] = ms.max()
                #ms = reconstruction(seed, ms, method='erosion')
                imgs[i,] = resize(tmp_img,(self.target_size[0],self.target_size[1])).reshape(*self.target_size,self.input_channels)
                masks[i,] = resize(tmp_msk,(self.target_size[0],self.target_size[1])).reshape(*self.target_size,1)

        return imgs, masks
		
	def __init_info(self):
        s = {"train" : "Training",
            "augmented" : "Training",
            "val" : "Validating",
            "test" : "Testing"}
        print(s[self.mode] + " on " + str(len(self.image_list)) + " Files")
    
# Option1: Keras Sequence Class:
params = {'target_size': SHAPE,
          'batch_size': 4,
          'shuffle': True}

# Generators
training_generator = DataGenerator(mode='augmented',**params)
validation_generator = DataGenerator(mode='val',**params)
