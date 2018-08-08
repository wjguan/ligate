import keras.backend as K
import tensorflow as tf
import numpy as np


def weighted_categorical_crossentropy(class_weights):
    '''
    Assigns weights to each category in categorical cross entropy 
    Inputs:
    class_weights (numpy ndarray) - 1D array of shape (C,) where C is the number of classes 
    '''
    weights = K.variable(class_weights)
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss,-1)
        return loss 
    return loss 