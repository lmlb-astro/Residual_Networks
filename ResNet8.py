import tensorflow as tf
from tensorflow.keras.layers import Flatten


## A Residual network with 8 layers
## Designed for input shape (128, 128, _)
class ResNet8(tf.keras.Model):
    
    def __init__(self, num_out, filters = [16, 32, 64, 128], drop_rate = 0.25, act_out = 'softmax', fl_type = 'float32', **kwargs):
        super(ResNet8, self).__init__(**kwargs)
        
        ## verify the number of filters provided
        if len(filters) != 4: raise AttributeError('The length of the filters list has to be 4, containing only integers.')
        
        ## initialize the first convolutional layer, batch normalization and max pool layer
        self.conv1 = tf.keras.layers.Conv2D(filters = filters[0], kernel_size = (5,5), strides = (1, 1), padding = 'SAME')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.mp1 = tf.keras.layers.MaxPool2D(pool_size = (2, 2)) ## gives: (64, 64, filters[0])
        
        ## initialize the residual blocks with max pooling
        self.res1 = ResBlock2(filters = filters[1], kernel_size = (3, 3))
        self.mp2 = tf.keras.layers.MaxPool2D(pool_size = (2, 2)) ## gives (32, 32, filters[1])
        
        self.res2 = ResBlock2(filters = filters[2], kernel_size = (3, 3))
        self.mp3 = tf.keras.layers.MaxPool2D(pool_size = (2, 2)) ## gives (16, 16, filters[2])
        
        self.res3 = ResBlock2(filters = filters[3], kernel_size = (3, 3))
        self.mp4 = tf.keras.layers.MaxPool2D(pool_size = (2, 2)) ## gives (8, 8, filters[3])
        
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
        
        x = self.res2(x)
        x = self.mp3(x)
        
        x = self.res3(x)
        x = self.mp4(x)
        
        ## dropout and flatten
        x = self.drop1(x)
        x = self.flatten(x)
        
        ## feed-forward
        return self.d1(x)