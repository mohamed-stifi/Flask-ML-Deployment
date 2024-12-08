import keras
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

def vgg16_predict(img_path, model):
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)

    label = decode_predictions(features)
    label = label[0][0]
    return label