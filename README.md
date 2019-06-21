# UNet for Rumex Obtusifolius segmentation and root-detection in 2D field-images

![alt text](assets/segmentation.png)

Prediction: Yellow Mask, Ground Truth: Cyan Border

![alt text](assets/roots.png)

Prediction: Black Point, Ground Truth: Yellow Point

#### Required Packages:
- tensorflow
- keras
- pandas
- numpy
- imgaug
- seaborn
- skimage
- pil
- sklearn
- matplotlib
- cv2
- imutils
 
 # Notebooks:
 * [10_inspect_data](notebooks/10_inspect_data.ipynb) Resizies the images from "LabelingTool" in in same sizes including the labels (mask, root-centers, leaf-splines)
 * [20_training](notebooks/20_training.ipynb) Training Process of the Unet
 * [30_results](notebooks/30_results.ipynb) Visualize the optical Results and Evaluate both Segmentation and Root-Detection
 * [40_layer_visualization](notebooks/40_layer_visualization.ipynb) Visualize the intermediate Layers of the trained Network
 * [50_weights_visualization](notebooks/50_weights_visualization.ipynb) Visualize the weights of the Convolution-Kernels of both trained and untrained model
