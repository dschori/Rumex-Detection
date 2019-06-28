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

"""
   Visualize constructor

   :param "pandas dataframe" df: dataframe
   :param tuple input_shape: Input image shape (height, width, colorchannels)
   :param str masktype: Which masktype to load ("leaf" or "root")
   :param str predictiontype: Which prediction to make ("leaf" or "root")
   :param "Keras Model" model: Trained Keras Model
   :return: image
"""
```

#### Functions:
```python

Visualize.get_image(index)

"""
   Get Image of selected index from Dataframe specified

   :param int or str "index": index of Image to load
   :return: image
"""
```
___
```python
Visualize.get_mask(index)

"""
   Get Mask of selected index from Dataframe specified

   :param int or str "index": index of Mask to load
   :return: image
"""
```
___
```python
Visualize.get_roots(index)

"""
   Get Roots Coordinates of selected index from Dataframe specified

   :param int or str "index": index of Image to load roots from
   :return: Root Coordinates as List of Tuples (x,y)
"""
```
___
```python
Visualize.get_prediction(index)

"""
   Get Prediction of Image of selected index from Dataframe specified

   :param int or str "index": index of Image to make prediction
   :return: Prediction as Image
"""
```
___
```python
Visualize.show_single(index, mode)

"""
   Show Single Image of selected index from Dataframe specified

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
   Show a rows x 2 Matrix of images of selected indexes (or random) from Dataframe specified

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


### Evaluate(Visualize) (class):
#### Init:
Example:
```python
ev_params = {'df':val20_df,
              'input_shape':(512,768,3),
              'masktype':'leaf',
              'predictiontype':'root',
              'model':model}

ev = Evaluate(**ev_params)

"""
   Evaluate constructor

   :param "pandas dataframe" df: dataframe
   :param tuple input_shape: Input image shape (height, width, colorchannels)
   :param str masktype: Which masktype to load ("leaf" or "root")
   :param str predictiontype: Which prediction to make ("leaf" or "root")
   :param "Keras Model" model: Trained Keras Model
   :return: image
"""
```

#### Functions:
```python

Evaluate.get_seg_eval_metrics(index)

"""
   Returns Segmentation Evaluation Scores for leaf segmentation
   
   :param float prediction_threshold: Threshold applied to prediciton mask
   :param float dc_threshold: Dice Coeff Threshold bottom level -> Predictions lower are classified as False Negatives 
   :return: DCs, TPs, FPs, FNs, names (All Dice Coefficients, True Positives, False Positives, False Negatives and Names as List)
"""
```
___
```python
Evaluate.get_dice_score(index)

"""
   Calculates Dice Score of selected index

   :param int or str "index": index of Mask to load
   :param float prediction_threshold: Threshold applied to prediciton mask
   :return: Dice_Coefficient as Float
"""
```
___
```python
Evaluate.get_iou_score(index)

"""
   Calculates Intersection over Union of selected Index

   :param int or str "index": index of Mask to load
   :param float prediction_threshold: Threshold applied to prediciton mask
   :return: IOU_Score as Float
"""
```
___
```python
Evaluate.get_root_pred_coord_v1(index)

"""
   Calculates Root Positions

   :param int or str "index": index of Mask to load
   :param float prediction_threshold: Threshold applied to prediciton mask
   :return: roots_pred as list of tuples (X,Y)
"""
```
___
```python
Evaluate.get_root_precicion(index)

"""
   Calculates Precision values for Root Detection of a single Index

   :param int or str "index": index of Mask to load
   :param int tolerance: tolerance radius in pixels
   :return: tP, fP, fN, precision, recall (True Positve, False Positive, False Negative, Precision and Recall Scores)
"""
```
___
