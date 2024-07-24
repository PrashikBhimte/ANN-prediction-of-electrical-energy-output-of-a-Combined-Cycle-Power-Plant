import tensorflow as tf
import numpy as np

ann = tf.keras.models.load_model('./model.keras')

pred = ann.predict(np.array([[8.34, 40.77, 1010.84, 90.01]]))

print("Prediction : ", pred)