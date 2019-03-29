from keras.models import Model
from keras.layers import Input, concatenate, MaxPooling2D,Conv2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop, Adadelta

from imports.models.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff, iou, iou_loss
import numpy as np

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
        model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff,iou])
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
    down = Conv2D(n_feature_maps, (3, 3), padding='same')(input)
    down = BatchNormalization()(down)
    down = Activation('relu')(down)
    down = Conv2D(n_feature_maps, (3, 3), padding='same')(down)
    down = BatchNormalization()(down)
    down = Activation('relu')(down)
    concat_layer = down
    down = MaxPooling2D((2, 2), strides=(2, 2))(down)
    return down, concat_layer

def up_block(input,concat_layer,n_feature_maps):
    up = UpSampling2D((2, 2))(input)
    up = concatenate([concat_layer, up], axis=3)
    up = Conv2D(n_feature_maps, (3, 3), padding='same')(up)
    up = BatchNormalization()(up)
    up = Activation('relu')(up)
    up = Conv2D(n_feature_maps, (3, 3), padding='same')(up)
    up = BatchNormalization()(up)
    up = Activation('relu')(up)
    up = Conv2D(n_feature_maps, (3, 3), padding='same')(up)
    up = BatchNormalization()(up)
    up = Activation('relu')(up)
    return up

def get_unet_mod(input_shape=(1024, 1024, 3),num_classes=1):
    concats_list = []
    feature_maps = [8,16,32,64,128,256,512] #without center
    input = Input(shape=input_shape)
    origin = input
    
    # downsampling:
    for d in range(len(feature_maps)):
        output, concat_layer = down_block(input,feature_maps[d])
        input = output
        concats_list.append(concat_layer)

    # center
    center = Conv2D(feature_maps[-1]*2, (3, 3), padding='same')(output)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(feature_maps[-1]*2, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    input = center
    print(concats_list[0])
    # upsampling:
    for u in range(len(feature_maps)):
        ouput = up_block(input,concats_list[::-1][u],feature_maps[::-1][u])
        input = output

    if num_classes == 1:
        map1 = Conv2D(1, (1, 1), activation='sigmoid',name="map1")(output)
        model = Model(inputs=origin, outputs=map1)
        model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff,iou])
        return model
    elif num_classes == 2:
        map1 = Conv2D(1, (1, 1), activation='sigmoid',name="map1")(output)
        map2 = Conv2D(1, (1, 1), activation='sigmoid',name="map2")(output)
        model = Model(inputs=origin, outputs=[map1, map2],)
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


def get_unet_pretrained(input_shape=(1024, 1024, 3),num_classes=1):
    from keras.applications.vgg16 import VGG16 as PTModel
    base_pretrained_model = PTModel(input_shape =  (512,768,3), include_top = False, weights = 'imagenet')
    base_pretrained_model.trainable = False

    from collections import defaultdict, OrderedDict
    from keras.models import Model
    layer_size_dict = defaultdict(list)
    inputs = []
    for lay_idx, c_layer in enumerate(base_pretrained_model.layers):
        if not c_layer.__class__.__name__ == 'InputLayer':
            layer_size_dict[c_layer.get_output_shape_at(0)[1:3]] += [c_layer]
        else:
            inputs += [c_layer]
    # freeze dict
    layer_size_dict = OrderedDict(layer_size_dict.items())
    for k,v in layer_size_dict.items():
        print(k, [w.__class__.__name__ for w in v])

    # take the last layer of each shape and make it into an output
    pretrained_encoder = Model(inputs = base_pretrained_model.get_input_at(0), 
                            outputs = [v[-1].get_output_at(0) for k, v in layer_size_dict.items()])
    pretrained_encoder.trainable = False
    n_outputs = pretrained_encoder.predict([np.zeros((1,512,768,3))])

    from keras.layers import Input, Conv2D, concatenate, UpSampling2D, BatchNormalization, Activation, Cropping2D, ZeroPadding2D
    x_wid, y_wid = (512,768)
    in_t0 = Input((512,768,3), name = 'T0_Image')
    wrap_encoder = lambda i_layer: {k: v for k, v in zip(layer_size_dict.keys(), pretrained_encoder(i_layer))}

    t0_outputs = wrap_encoder(in_t0)
    lay_dims = sorted(t0_outputs.keys(), key = lambda x: x[0])
    skip_layers = 2
    last_layer = None
    for k in lay_dims[skip_layers:]:
        cur_layer = t0_outputs[k]
        channel_count = cur_layer._keras_shape[-1]
        cur_layer = Conv2D(channel_count//2, kernel_size=(3,3), padding = 'same', activation = 'linear')(cur_layer)
        cur_layer = BatchNormalization()(cur_layer) # gotta keep an eye on that internal covariant shift
        cur_layer = Activation('relu')(cur_layer)
        
        if last_layer is None:
            x = cur_layer
        else:
            last_channel_count = last_layer._keras_shape[-1]
            x = Conv2D(last_channel_count//2, kernel_size=(3,3), padding = 'same')(last_layer)
            x = UpSampling2D((2, 2))(x)
            x = concatenate([cur_layer, x])
        last_layer = x
    final_output = Conv2D(1, kernel_size=(1,1), padding = 'same', activation = 'sigmoid')(last_layer)
    crop_size = 20
    final_output = Cropping2D((crop_size, crop_size))(final_output)
    final_output = ZeroPadding2D((crop_size, crop_size))(final_output)
    unet_model = Model(inputs = [in_t0],
                    outputs = [final_output])

    import keras.backend as K
    from keras.optimizers import Adam
    from keras.losses import binary_crossentropy
    def dice_coef(y_true, y_pred, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
    def dice_p_bce(in_gt, in_pred):
        return 0.0*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)
    def true_positive_rate(y_true, y_pred):
        return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

    unet_model.compile(optimizer=Adam(1e-3, decay = 1e-6), 
                    loss=dice_p_bce, 
                    metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
    return unet_model