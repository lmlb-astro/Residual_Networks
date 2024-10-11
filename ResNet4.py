import tensorflow as tf
import keras
from tensorflow.keras.layers import Flatten

import Residual_block2 as resbl




## A Residual network with 4 layers
@keras.saving.register_keras_serializable(package="MyLayers")
class ResNet4(tf.keras.Model):
    
    def __init__(self, num_out, filters = [16, 32], drop_rate = 0.25, act_out = 'softmax', fl_type = 'float32', **kwargs):
        super(ResNet4, self).__init__(**kwargs)
        
        ## verify the number of filters provided
        if len(filters) != 2: raise AttributeError('The length of the filters list has to be 2, containing only integers.')
        
        ## store the input parameters
        self.num_out = num_out
        self.filters = filters
        self.drop_rate = drop_rate
        self.act_out = act_out
        self.fl_type = fl_type
        
        ## initialize the first convolutional layer, batch normalization and max pool layer
        self.conv1 = tf.keras.layers.Conv2D(filters = self.filters[0], kernel_size = (7, 7), strides = (1, 1), padding = 'SAME')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.mp1 = tf.keras.layers.MaxPool2D(pool_size = (2, 2))
        
        ## initialize the residual blocks with max pooling
        self.res1 = resbl.ResBlock2(filters = self.filters[1], kernel_size = (3, 3))
        self.mp2 = tf.keras.layers.MaxPool2D(pool_size = (2, 2))
        
        ## dropout layer
        self.drop1 = tf.keras.layers.Dropout(self.drop_rate)
        
        ## flatten and final dense layers
        self.flatten = Flatten(dtype = self.fl_type)
        self.d1 = tf.keras.layers.Dense(self.num_out, activation = self.act_out)
        
        
        
    def get_config(self):
        base_config = super().get_config()
        sub_config = {"num_out": self.num_out, 
                      "filters": self.filters,
                      "drop_rate": self.drop_rate,
                      "act_out": self.act_out,
                      "fl_type": self.fl_type
                     }
        
        return {**base_config, **sub_config}
        
        
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