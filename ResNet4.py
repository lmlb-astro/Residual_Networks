import tensorflow as tf
from tensorflow.keras.layers import Flatten

import Residual_block2 as resbl




## A Residual network with 4 layers
class ResNet4(tf.keras.Model):
    
    def __init__(self, num_out, filters = [16, 32], drop_rate = 0.25, act_out = 'softmax', fl_type = 'float32', **kwargs):
        super(ResNet4, self).__init__(**kwargs)
        
        ## verify the number of filters provided
        if len(filters) != 2: raise AttributeError('The length of the filters list has to be 2, containing only integers.')
        
        ## initialize the first convolutional layer, batch normalization and max pool layer
        self.conv1 = tf.keras.layers.Conv2D(filters = filters[0], kernel_size = (5,5), strides = (1, 1), padding = 'SAME')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.mp1 = tf.keras.layers.MaxPool2D(pool_size = (2, 2))
        
        ## initialize the residual blocks with max pooling
        self.res1 = resbl.ResBlock2(filters = filters[1], kernel_size = (3, 3))
        self.mp2 = tf.keras.layers.MaxPool2D(pool_size = (2, 2))
        
        ## dropout layer
        self.drop1 = tf.keras.layers.Dropout(drop_rate)
        
        ## flatten and final dense layers
        self.flatten = Flatten(dtype = fl_type)
        self.d1 = tf.keras.layers.Dense(num_out, activation = act_out)
        
        
    ## call on the model
    def call(self, inputs):
        ## first convolutional layer
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.mp1(x)
        
        ## the 3 residual blocks
        x = self.res1(x)
        x = self.mp2(x)
        
        ## dropout and flatten
        x = self.drop1(x)
        x = self.flatten(x)
        
        ## feed-forward
        return self.d1(x)