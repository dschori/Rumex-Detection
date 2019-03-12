from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Dropout, Conv2DTranspose
from keras.optimizers import RMSprop

from imports.models.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff


def dense_block(input,num_layers,k_feature_maps):
    l_out = input
    c_out = input
    l_outs = []
    for _ in range(num_layers):
        l_out = layer(l_out,k_feature_maps)
        c_out = concatenate([c_out, l_out], axis=3)
        l_outs.append(l_out)
        l_out = c_out
    c_out = concatenate(l_outs, axis=3)
    output = c_out
    return output

def layer(input,k_feature_maps):
    output = BatchNormalization()(input)
    output = Activation('relu')(output)
    output = Conv2D(k_feature_maps,(3,3),padding='same')(output)
    #output = Dropout(0.2)(output)
    return output

def transition_down(input):
    output = BatchNormalization()(input)
    output = Activation('relu')(output)
    output = Conv2D(1,(3,3),padding='same')(output)
    #output = Dropout(0.2)(output)
    output = MaxPooling2D((2, 2), strides=(2, 2))(output)
    return output

def transition_up(input):
    #output = Conv2DTranspose(1,(3,3),padding='same',strides=(2,2))(input)
    output = UpSampling2D((2,2))(input)
    return output

def get_fc_dense_net(input_shape=(1024, 1024, 3),num_classes=1):

    input = Input(shape=input_shape)
    
    down1 = Conv2D(48,(3,3),padding='same')(input)

    down2 = dense_block(down1,num_layers=4,k_feature_maps=16)
    down2c = concatenate([down2, down1], axis=3)
    down2 = transition_down(down2c)

    down3 = dense_block(down2,num_layers=5,k_feature_maps=16)
    down3c = concatenate([down3, down2], axis=3)
    down3 = transition_down(down3c)

    down4 = dense_block(down3,num_layers=7,k_feature_maps=16)
    down4 = transition_down(down4)

    center = dense_block(down4,num_layers=10,k_feature_maps=16)

    up1 = transition_up(center)
    up1 = dense_block(up1,num_layers=7,k_feature_maps=16)

    up2 = transition_up(up1)
    up2 = dense_block(up2,num_layers=5,k_feature_maps=16)

    up3 = transition_up(up2)
    up3 = dense_block(up3,num_layers=4,k_feature_maps=16)


    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up3)

    model = Model(inputs=input, outputs=classify)

    model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    return model