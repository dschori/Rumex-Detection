from imgaug import augmenters as iaa
import skimage
from skimage.transform import rescale, resize

MODEL_INPUT_SHAPE = (512,768)

seq_norm = iaa.Sequential([
            iaa.CLAHE(),
            iaa.LinearContrast(alpha=1.0)])

def normalize(image):
    image = resize(image,MODEL_INPUT_SHAPE)
    image = (image*255).astype("uint8")
    image = seq_norm.augment_image(image)
    image = image.astype(float)/255.0
    return image

def calculate_coordinates(root_estimation):
        labels = skimage.measure.label(root_estimation)
        roots_pred = skimage.measure.regionprops(labels)
        roots_pred = [r for r in roots_pred if r.area > 1500]
        roots_pred = [r.centroid for r in roots_pred]
        roots_pred = [list(p) for p in roots_pred] #Convert to same format
        roots_pred = [p[::-1] for p in roots_pred] #Flipp X,Y
        return roots_pred

def predict(image, model):
    image = normalize(image)
    prediction = model.predict(image.reshape(1,*MODEL_INPUT_SHAPE,3)).reshape(*MODEL_INPUT_SHAPE,2)
    return prediction

def get_segmentation(image, model, threshold=0.8):
    prediction = predict(image, model)
    segmentation = prediction[:,:,0] > threshold
    return segmentation

def get_coordinates(image, model, threshold=0.8):
    prediction = predict(image, model)
    root_estimation = prediction[:,:,1] > threshold
    roots_list = calculate_coordinates(root_estimation)
    return roots_list

