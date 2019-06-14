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

## Simple UNet:
def get_unet(input_shape=(1024, 1024, 3),num_classes=1):

    assert num_classes == 1 or num_classes == 2 , "Number of output_classes not Supportet"

    inputs = Input(shape=input_shape)
    # 1024

    down0b = Conv2D(8, (3, 3), padding='same')(inputs)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b = Conv2D(8, (3, 3), padding='same')(down0b)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(down0b_pool)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(8, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    # 1024

    if num_classes == 1:
        map1 = Conv2D(1, (1, 1), activation='sigmoid',name="map1")(up0b)
        model = Model(inputs=inputs, outputs=map1)
        model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff,keras.losses.binary_crossentropy])
        return model
    elif num_classes == 2:
        map1 = Conv2D(1, (1, 1), activation='sigmoid',name="map1")(up0b)
        map2 = Conv2D(1, (1, 1), activation='sigmoid',name="map2")(up0b)
        model = Model(inputs=inputs, outputs=[map1, map2],)
        losses = {
        "map1": bce_dice_loss,
        "map2": "binary_crossentropy"}   
        lossWeights = {"map1": 2.0, "map2": 1.0}
        metrics = {
            "map1" : dice_coeff,
            "map2" : iou,
            "bce" : "binary_crossentropy"}
        model.compile(optimizer=RMSprop(lr=0.0001), loss=losses, loss_weights=lossWeights, metrics=metrics)
        return model
	
### TODO: Implement Unet as proposed in: https://arxiv.org/abs/1505.04597

def down_block(input,n_feature_maps):
    down = input
    for _ in range(2):
        down = Conv2D(n_feature_maps, (3, 3), padding='same')(down)
        down = BatchNormalization()(down)
        down = Activation('relu')(down)

    concat_layer = down
    down = MaxPooling2D((2, 2), strides=(2, 2))(down)
    return down, concat_layer

def up_block(input,concat_layer,n_feature_maps):
    up = UpSampling2D((2, 2))(input)
    up = concatenate([concat_layer, up], axis=3)
    for _ in range(3):
        up = Conv2D(n_feature_maps, (3, 3), padding='same')(up)
        up = BatchNormalization()(up)
        up = Activation('relu')(up)
    return up

def get_unet_mod(input_shape=(1024, 1024, 3),
                feature_maps=[16,32,64,128,256],
                num_classes=1):
    concats_list = []
    input = Input(shape=input_shape)
    origin = input
    input = CoordinateChannel2D()(input)
    
    # downsampling:
    for f in feature_maps:
        output, concat_layer = down_block(input,f)
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
    for f,c in zip(feature_maps[::-1],concats_list[::-1]):
        output = up_block(input,c,f)
        input = output

    final_layer = Conv2D(num_classes, (3, 3), padding='same')(output)
    final_layer = Activation("sigmoid")(final_layer)
    model = Model(inputs=origin, outputs=final_layer)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff,"bce"])
    return model    

class UNet():
    def __init__(self):
        self.model = None
        self.batchnorm = None
        self.encoder_type = None

    def get_model(self):
        return self.model

    def __compile_model(self):
        self.model.compile(optimizer=Adam(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff,iou_score])

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

    def __decoder_block(self,input,concat_layer,n_feature_maps):
        up = UpSampling2D((2,2))(input)
        up = concatenate([concat_layer,up],axis=3)
        for _ in range(2):
            up = Conv2D(n_feature_maps,(3,3),padding='same', activation='relu')(up)
            up = BatchNormalization()(up)
        return up

    def __encoder_block(self,input,n_feature_maps,names):
        down = input
        for n in names:
            down = Conv2D(n_feature_maps, (3, 3), padding='same', activation='relu',name=n)(down)
            down = BatchNormalization()(down) if self.batchnorm == True else down

        concat_layer = down
        down = MaxPooling2D((2, 2), strides=(2, 2))(down)
        return down, concat_layer

    def create_model(self,batch_norm=True,input_shape=(512,768,3),feature_maps=[16,32,64,128,256,512],num_classes=2):
        self.batchnorm = batch_norm
        concats_list = []
        input = Input(shape=input_shape)
        origin = input
        
        # downsampling:
        for f in feature_maps:
            output, concat_layer = self.__encoder_block(input,f,(str(f)+"a",str(f)+"b"))
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
        for f,c in zip(feature_maps[::-1],concats_list[::-1]):
            output = up_block(input,c,f)
            input = output

        final_layer = Conv2D(num_classes, (3, 3), padding='same')(output)
        final_layer = Activation("sigmoid")(final_layer)
        self.model = Model(inputs=origin, outputs=final_layer)
        self.__compile_model


    def create_pretrained_model(self,encoder_type='vgg19',batchnorm=True,coord_conv=True,input_shape=(512, 768, 3),num_classes=2):
        
        
        self.encoder_type = encoder_type
        self.batchnorm = batchnorm
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

            #concats_list.append(encoder_pretrained.get_layer('conv5_block3_out').output) #2048 16*24
            
            #center = MaxPooling2D((2, 2), strides=(2, 2))(encoder_pretrained.layers[-1].output)
            #center = Conv2D(2048, (3, 3), padding='same', activation='relu')(center)
            #center = BatchNormalization()(center)


        if encoder_type == "vgg19":
            encoder_pretrained = VGG19(include_top=False,weights='imagenet',input_tensor=input,input_shape=input_shape)
            concats_list.append(encoder_pretrained.get_layer('block1_conv2').output) #64, 512*768
            concats_list.append(encoder_pretrained.get_layer('block2_conv2').output) #128 256*384
            concats_list.append(encoder_pretrained.get_layer('block3_conv4').output) #256 128*192
            concats_list.append(encoder_pretrained.get_layer('block4_conv4').output) #512 64*96
            concats_list.append(encoder_pretrained.get_layer('block5_conv4').output) #512 32*48

            center = encoder_pretrained.layers[-1].output
            for _ in range(2):
                center = Conv2D(1024, (3, 3), padding='same', activation='relu')(center)
                center = BatchNormalization()(center)

        input = center

        # upsampling:
        if encoder_type == "resnet101":
            for f,c in zip([512,256,128,64],concats_list[::-1]):
                output = self.__decoder_block(input,c,f)
                input = output

            output = UpSampling2D((2,2))(output)
            output = Conv2D(32, (3, 3), padding='same', activation='relu')(output)

        if encoder_type == "vgg19":
            for f,c in zip([512,512,256,128,64],concats_list[::-1]):
                output = self.__decoder_block(input,c,f)
                input = output

        final_layer = Conv2D(num_classes, (3, 3), padding='same')(output)
        final_layer = Activation("sigmoid")(final_layer)
        self.model = Model(inputs=origin, outputs=final_layer)
        self.__compile_model()