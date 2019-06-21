# tools.py:

Functions to Visualize and Evaluate Images and Results.

## Usage:

### Visualize (class):

#### Init:
Example:
```python
vis_params = {'df':val20_df,
              'input_shape':(512,768,3),
              'masktype':'leaf',
              'predictiontype':'root',
              'model':model}

vis = Visualize(**vis_params)
```

#### Functions:
```python

Visualize.get_image(index)

"""
   Get Image

   :param int or str "index": index of Image to load
   :return: image
"""
```
___
```python
Visualize.get_mask(index)

"""
   Get Mask

   :param int or str "index": index of Mask to load
   :return: image
"""
```
___
```python
Visualize.get_roots(index)

"""
   Get Roots Coordinates

   :param int or str "index": index of Image to load roots from
   :return: Root Coordinates as List of Tuples (x,y)
"""
```
___
```python
Visualize.get_prediction(index)

"""
   Get Prediction of Image

   :param int or str "index": index of Image to make prediction
   :return: Prediction as Image
"""
```
___
```python
Visualize.show_single(index, mode)

"""
   Show Single Image

   :param int or str "index": index of Image to show
   :param str "mode": 
        image : shows only image
        mask : shows only mask
        image_mask : shows image with overlayed mask
        image_prediction : shows image with overlayed prediction
        image_prediction_roots : shows image with GT mask and predicted roots
        image_prediction_contour : shows image with predicted segmentation and GT contours
   :return: No return Value
"""
```
___
```python
Visualize.show_matrix(index, mode, rows=4)

"""
   Show a rows x 2 Matrix of images

   :param List of int or str: List of indexes to show, or "random"
   :param str "mode": 
        image : shows only image
        mask : shows only mask
        image_mask : shows image with overlayed mask
        image_prediction : shows image with overlayed prediction
        image_prediction_roots : shows image with GT mask and predicted roots
        image_prediction_contour : shows image with predicted segmentation and GT contours
   :param int "row": how much rows should be displayd
   :return: No return Value
"""
```
___


### Evaluate (class):
