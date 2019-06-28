from keras.models import Model
from keras.layers import Input, concatenate, MaxPooling2D,Conv2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop, Adadelta, SGD, Adam
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from unet.coord import *

from unet.losses import *

import numpy as np

from keras_applications.resnet import ResNet101
import keras
import keras_applications
keras_applications.set_keras_submodules(
    backend=keras.backend,
    layers=keras.layers,
    models=keras.models,
    utils=keras.utils
)

class UNet():
    def __init__(self):
        self.model = None
        self.batch_norm = None
        self.encoder_type = None

    def get_model(self):
        return self.model

    def create_model(self,batch_norm=True,input_shape=(512,768,3),feature_maps=[16,32,64,128,256],num_classes=2):
        self.batch_norm = batch_norm
        concats_list = []
        input = Input(shape=input_shape)
        origin = input
        
        # downsampling:
        i=0
        for f in feature_maps:
            i+=1
            block_name = 'encoder_block' + str(i) + "_conv"
            output, concat_layer = self.__encoder_block(input,f,block_name)
            input = output
            concats_list.append(concat_layer)

        # center
        center = input 
        for _ in range(2):
            center = Conv2D(feature_maps[-1]*2, (3, 3), padding='same')(center)
            center = BatchNormalization()(center)
            center = Activation('relu')(center)

        input = center
        # upsampling:
        i=0
        for f,c in zip(feature_maps[::-1],concats_list[::-1]):
            i+=1
            block_name = 'decoder_block' + str(i) + "_conv"
            output = self.__decoder_block(input,c,f,block_name)
            input = output

        final_layer = Conv2D(num_classes, (3, 3), padding='same')(output)
        final_layer = Activation("sigmoid")(final_layer)
        self.model = Model(inputs=origin, outputs=final_layer)
        self.__compile_model


    def create_pretrained_model(self,encoder_type='vgg19',batch_norm=False,coord_conv=False,input_shape=(512, 768, 3),num_classes=2):        
        self.encoder_type = encoder_type
        self.batch_norm = batch_norm
        concats_list = []
        input = Input(shape=input_shape)
        origin = input
        input = CoordinateChannel2D()(input) if coord_conv == True else input

        if encoder_type == "resnet101":
            encoder_pretrained = ResNet101(include_top=False,weights='imagenet',input_tensor=input,input_shape=input_shape)
            concats_list.append(encoder_pretrained.get_layer('conv1_relu').output) #64, 256*384
            concats_list.append(encoder_pretrained.get_layer('conv2_block3_out').output) #256 128*192
            concats_list.append(encoder_pretrained.get_layer('conv3_block4_out').output) #512 64*96
            concats_list.append(encoder_pretrained.get_layer('conv4_block23_out').output) #1024 32*48
            center = encoder_pretrained.layers[-1].output

        if encoder_type == "vgg19":
            encoder_pretrained = VGG19(include_top=False,weights='imagenet',input_tensor=input,input_shape=input_shape)
            concats_list.append(encoder_pretrained.get_layer('block1_conv2').output) #64, 512*768
            concats_list.append(encoder_pretrained.get_layer('block2_conv2').output) #128 256*384
            concats_list.append(encoder_pretrained.get_layer('block3_conv4').output) #256 128*192
            concats_list.append(encoder_pretrained.get_layer('block4_conv4').output) #512 64*96
            concats_list.append(encoder_pretrained.get_layer('block5_conv4').output) #512 32*48

            center = encoder_pretrained.layers[-1].output
            for _ in range(1):
                center = Conv2D(1024, (3, 3), padding='same', activation='relu')(center)
                center = BatchNormalization()(center)

        input = center

        # upsampling:
        i = 0
        if encoder_type == "resnet101":
            for f,c in zip([512,256,128,64],concats_list[::-1]):
                i+=1
                block_name = 'decoder_block' + str(i) + "_conv"
                output = self.__decoder_block(input,c,f,block_name)
                input = output

            output = UpSampling2D((2,2))(output)
            output = Conv2D(32, (3, 3), padding='same', activation='relu')(output)

        if encoder_type == "vgg19":
            for f,c in zip([256,128,64,32,16],concats_list[::-1]):
                i+=1
                block_name = 'decoder_block' + str(i) + "_conv"
                output = self.__decoder_block(input,c,f,block_name)
                input = output

        final_layer = Conv2D(num_classes, (3, 3), padding='same')(output)
        final_layer = Activation("sigmoid")(final_layer)
        self.model = Model(inputs=origin, outputs=final_layer)
        self.__compile_model()

    def freeze_encoder(self,model,encoder_type):
        self.model = model

        for layer in self.model.layers:
            layer.trainable = False
            if layer.name == "conv5_block3_out" and encoder_type == "resnet101":
                break
            if layer.name == "block5_pool" and encoder_type == "vgg19":
                break
        self.__compile_model()

    def unfreeze_encoder(self,model,encoder_type):
        self.model = model

        for layer in self.model.layers:
            layer.trainable = True
            if layer.name == "conv5_block3_out" and encoder_type == "resnet101":
                break
            if layer.name == "block5_pool" and encoder_type == "vgg19":
                break
        self.__compile_model()

    def __decoder_block(self,input,concat_layer,n_feature_maps,block_name):
        up = UpSampling2D((2,2))(input)
        up = concatenate([concat_layer,up],axis=3)
        for i in range(2):
            up = Conv2D(n_feature_maps,(3,3),padding='same', activation='relu',name=block_name+str(i+1))(up)
            up = BatchNormalization()(up)
        return up

    def __encoder_block(self,input,n_feature_maps,block_name):
        down = input
        for i in range(2):
            down = Conv2D(n_feature_maps, (3, 3), padding='same', activation='relu',name=block_name+str(i+1))(down)
            down = BatchNormalization()(down) if self.batch_norm == True else down

        concat_layer = down
        down = MaxPooling2D((2, 2), strides=(2, 2))(down)
        return down, concat_layer

    def __compile_model(self):
        self.model.compile(optimizer=Adam(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff,iou_score])
