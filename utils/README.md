# tools.py

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
