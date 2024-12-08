import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import numpy as np

model = VGG16(weights='imagenet')

print(model.summary())
print('-'*50)


img_path = 'elephant.jpg'
img = keras.utils.load_img(img_path, target_size=(224, 224))
x = keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)


label = decode_predictions(features)
label = label[0][0]

print('%s (%.2f%%)' % (label[1], label[2]*100))
