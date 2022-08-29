"""
Usage:
python -m breed_predicter <path_to_img_file>

It returns the predicted breed.

Modules requirement:
-numpy
-tensorflow
-skimage

"""

import numpy as np
import tensorflow as tf
import pickle
from skimage import exposure
import sys

def select_dim(i,s):
  tmp = np.zeros(shape=s)
  tmp[:,:,i] = 1.0
  return tmp

def equalize(img):
    #print(f"shape: {img.shape} {type(img)}")
    s = img.shape
    for i in range(3):
      img[:,:,i] = exposure.equalize_hist(img,mask=select_dim(i,s))[:,:,i]
    return img

@tf.function (input_signature=[tf.TensorSpec(shape=(None,None,3), dtype=tf.float32)])
def tf_equalize(img):
  with tf.init_scope():
    return equalize(img.numpy())

class Equalize(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
      super(Equalize, self).__init__(**kwargs) #dynamic=True)

    def call(self, inputs,train=False):
        return tf_equalize(inputs)
    
    def compute_output_signature(self,input_signature):
      return input_signature


class PredictBreed:

  def __init__(self,model_path,classes_path):
      self.model   = tf.keras.models.load_model(model_path,custom_objects={'Equalize':Equalize})
      with open(str(classes_path), "rb") as input_file:
        self.class_names = pickle.load(input_file)
      self.preprocess = tf.keras.models.Sequential(
          name = "preprocess",
          layers = [   
            tf.keras.layers.Resizing(*IMG_SIZE,crop_to_aspect_ratio=False),
            #tf.keras.layers.Rescaling(scale=(1./255))  
          ]
        )  

  def predict(self,img_path):
    img      = tf.keras.preprocessing.image.load_img(img_path)
    np_img   = tf.keras.preprocessing.image.img_to_array(img)
    #simulate batch dim
    input      = np.expand_dims(np_img, axis=0)
    # predict with model 
    preds      = self.model.predict(self.preprocess(input))
    #deduct class name
    pred_label = self.class_names[np.argmax(preds)]

    return pred_label
    

if (__name__ == '__main__'):
 
    # total arguments
    if (len(sys.argv) == 2)
        pred_model = PredictBreed(r'./my_pretrained_vgg1_20220712-102633_dev.h5',r'./models/class_names.save')
        return pred_model.predict(sys.argv)
    else:
        print("Usage: python -m breed_predicter <path_to_img_file>")
