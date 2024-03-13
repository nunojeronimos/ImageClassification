import tensorflow as tf
import os
import cv2
import numpy as np

from keras.models import  load_model

img = cv2.imread('test.jpg')
resize = tf.image.resize(img,(256, 256))
new_model = load_model(os.path.join('models','dogidentifier.h5'))
pic = new_model.predict(np.expand_dims(resize/255,0))

if pic > 0.5:
    print(f'Is a dog')
else:
    print(f'Is not a dog')