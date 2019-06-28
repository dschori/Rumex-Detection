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

Class to create a Data Generator to Train the Model.

## Usage:
Example for a datagenerator:
```python
params = {'target_size': Config.SHAPE,
          'batch_size': 4,
          'input_channels': 3,
          'shuffle': True,
          'output':"both"}

gen = DataGenerator(df=train80_df,hist_equal=True,augment_data=True,save_images=False,**params)
train_gen = iter(gen)

"""
   :param Pandas Dataframe "df": Dataframe to load data from
   :param tuple "target_size": Target Size of Images
   :param int "batch_size": How many images per Batch to return
   :param int "input_channels": Number of Input Channels of Image
   :param bool "shuffle": Whether to shuffle the data after each epoch or not
   :param str "output": Which mask to output ("leaf", "root", "both")
   :param bool "hist_equal": Whether to Equalize Histogramms if Images or not
   :param bool "augment_data": Whether to Augment during Training or not
   :return Generator
"""
```
