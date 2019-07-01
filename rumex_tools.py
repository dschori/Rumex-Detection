from imgaug import augmenters as iaa
import skimage
from skimage.transform import rescale, resize
from keras.models import load_model
from unet.losses import *

MODEL_INPUT_SHAPE = (512,768)

class Rumex_Detection():
        def __init__(self, model_path):
                self.model_path = model_path
                self.seq_norm = iaa.Sequential([
                                iaa.CLAHE(),
                                iaa.LinearContrast(alpha=1.0)])
                self.__load_model()
                self.img = None
                self.pred = None
                self.root_coords = None

        def __load_model(self):
                self.model = load_model(self.model_path, custom_objects={'bce_dice_loss': bce_dice_loss,'dice_coeff':dice_coeff,'iou_score':iou_score})

        def __normalize(self):
                self.img = resize(self.img, MODEL_INPUT_SHAPE)
                self.img = (self.img*255).astype("uint8")
                self.img = self.seq_norm.augment_image(self.img)
                self.img = self.img.astype(float)/255.0

        def __predict(self):
                self.pred = self.model.predict(self.img.reshape(1,*MODEL_INPUT_SHAPE,3)).reshape(*MODEL_INPUT_SHAPE,2)

        def __calc_coords_v1(self, threshold):
                labels = skimage.measure.label(self.pred[:,:,1]>threshold)
                root_coords = skimage.measure.regionprops(labels)
                root_coords = [r for r in root_coords if r.area > 500]
                root_coords = [r.centroid for r in root_coords]
                root_coords = [list(p) for p in root_coords] #Convert to same format
                root_coords = [p[::-1] for p in root_coords] #Flipp X,Y
                self.root_coords = root_coords

        def get_leaf_mask(self, image, threshold=0.8):
                self.img = image
                self.__normalize()
                self.__predict()
                return self.pred[:,:,0].reshape(MODEL_INPUT_SHAPE) > threshold
        
        def get_root_coords(self, image, threshold=0.5):
                self.img = image
                self.__normalize()
                self.__predict()
                self.__calc_coords_v1(threshold)
                return self.root_coords
