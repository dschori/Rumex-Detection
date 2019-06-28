# unet.py:

Class to create Unet-model.

## Usage:

Example for a pretrained Model:
```python
unet = UNet()
unet.create_pretrained_model(encoder_type="vgg19", input_shape=(512, 768, 3), num_classes=2)
model = unet.get_model()
```

#### Functions:
```python

UNet.create_model(index)

"""
   Creates untrained model

   :param bool "batchnorm": Add Batch Normalization Layers in Encoder Blocks or not
   :param tuble of int "input_shape" 
"""
```
___
```python

UNet.create_pretrained_model(index)

"""
   Creates (Encoder) Pretrained Model

   :param str "encoder_type": Which Encoder Type ("vgg19", or "resnet101")
   :param bool "batch_norm": Add Batch Normalization Layers in Encoder Blocks or not
   :param tuble of int "input_shape" 
"""
```
___
```python

UNet.freeze_encoder(index)

"""
   Freezes Encoder of Model

   :param str "encoder_type": Which Encoder Type ("vgg19", or "resnet101")
   :param Keras Model "model": Model to freeze the Encoder
"""
```
___
```python

UNet.unfreeze_encoder(index)

"""
   Unfreezes Encoder of Model

   :param str "encoder_type": Which Encoder Type ("vgg19", or "resnet101")
   :param Keras Model "model": Model to unfreeze the Encoder
"""
```
___
```python

UNet.get_model(index)

"""
   Returns the Model

   :return Keras Model "model": Compiled Keras Model
"""
```
___

# data_generator.py:

Class to create Unet-model.

## Usage:
