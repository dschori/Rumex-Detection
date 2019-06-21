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
   :return: image
"""
```
___
