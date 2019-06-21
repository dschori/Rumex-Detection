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

### Functions:

```python
Visualize.get_image(index)

"""
   Get Image

   :param int or str index: index of Image to load
   :return: image
"""
```
___
```python
Visualize.get_mask(index)

"""
   Get Mask

   :param int or str index: index of Mask to load
   :return: image
"""
```
___
```python
Visualize.get_roots(index)

"""
   Get Roots Coordinates

   :param int or str index: index of Image to load roots from
   :return: Root Coordinates as List of Tuples (x,y)
"""
```
___
