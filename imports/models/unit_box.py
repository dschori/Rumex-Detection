from keras.models import Model
from keras.layers import Input, concatenate, MaxPooling2D,Conv2D, Activation, UpSampling2D, BatchNormalization, Conv2DTranspose
from keras.optimizers import RMSprop, Adadelta
from keras.applications.vgg16 import VGG16
from keras.losses import binary_crossentropy
import tensorflow as tf
import keras.backend as K

from imports.models.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff, iou, iou_loss
import numpy as np

_EPSILON = 10e-8

def IOULoss(label, input):
    """
    :param input: the estimate position
    :param label: the ground truth position
    :return: the IoU loss
    """
    # the estimate position
    xt, xb, xl, xr = tf.split(input, num_or_size_splits=4, axis=3)
    #xt = input[:,:,:,0]
    #xb = input[:,:,:,1]
    #xl = input[:,:,:,2]
    #xr = input[:,:,:,3]

    # the ground truth position
    gt, gb, gl, gr = tf.split(label, num_or_size_splits=4, axis=3)
   #gt = label[:,:,:,0]
    #gb = label[:,:,:,1]
    #gl = label[:,:,:,2]
    #gr = label[:,:,:,3]

    # compute the bounding box size
    X = (xt + xb) * (xl + xr)
    G = (gt + gb) * (gl + gr)

    # compute the IOU
    Ih = K.minimum(xt, gt) + K.minimum(xb, gb)
    Iw = K.minimum(xl, gl) + K.minimum(xr, gr)

    I = tf.multiply(Ih, Iw, name="intersection")
    #I = Ih * Iw
    U = X + G - I + _EPSILON

    IoU = tf.divide(I, U, name='IoU')
    #IoU = I / U

    L = tf.where(tf.less_equal(gt, tf.constant(0.01, dtype=tf.float32)),
                 tf.zeros_like(xt, tf.float32),
                 -tf.log(IoU + _EPSILON))

    print(L)

    #return tf.reduce_mean(L)
    return -tf.log(IoU)

def loss_function(score_pred,score_true,bbox_pred,bbox_true):
    score_loss = binary_crossentropy(score_pred,score_true)

    bbox_loss = IOULoss(bbox_pred,bbox_true)

    l2 = 0.

    return 0.01*score_loss + bbox_loss + l2



def unit_box(input_shape=(1024, 1024, 3),num_classes=1):
    inputs = Input(shape=input_shape)

    encoder = VGG16(include_top=False,input_shape=input_shape)

    for layer in encoder.layers:
        layer.trainable = False
    
    pool4 = encoder.get_layer('block4_pool').output
    pool5 = encoder.get_layer('block5_pool').output

    score_conv = Conv2D(1, (3, 3), padding='same',activation='linear',name='score_conv')(pool4)

    score = Conv2DTranspose(1,(32,32),strides=(16,16),padding='same',name='score')(score_conv)

    prob = Activation('sigmoid',name='prob')(score)

    bbox_conv = Conv2D(4,(3,3),padding='same',activation='linear',name='bbox_conv')(pool5)

    bbox = Conv2DTranspose(4,(64,64),strides=(32,32),padding='same',name='bbox')(bbox_conv)
   

    model = Model(inputs = encoder.input, outputs = [score, bbox])

    losses = {
        "score":"binary_crossentropy",
        "bbox":IOULoss
    }

    lossWeights = {"score": 0.01, "bbox": 1.0}

    
    metrics = {
        "score" : "binary_crossentropy",
        "bbox" : IOULoss}

    model.compile(optimizer=RMSprop(lr=0.0001), loss=losses,loss_weights=lossWeights, metrics=metrics)

    return model