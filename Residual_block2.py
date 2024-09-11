import tensorflow as tf


## Class that creates Residual network block with 2 convolutional layers


## Residual block with 2 convolutional layers
class ResBlock2(tf.keras.layers.Layer):
    
    def __init__(self, filters = 4, kernel_size = (3, 3), use_bias = False, **kwargs):
        ## initialize the parent class
        super(ResBlock2, self).__init__(**kwargs)

        ## initialize the two convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = 'SAME', use_bias = use_bias)
        self.conv2 = tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = (1, 1), padding = 'SAME', use_bias = use_bias)

        ## initialize the batch normalization layers
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        ## initialize the ReLu layers
        self.relu1 = tf.keras.layers.ReLU()
        self.relu2 = tf.keras.layers.ReLU()

        ## add the elementwise addition layer to complete the residual block
        self.add_layer = tf.keras.layers.Add()
    

    
    ## call ResBlock2()
    def call(self, inputs):
        ## first convolutional layer + normalization + activation
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        
        ## second convolutional layer + normalization
        x = self.conv2(x)
        x = self.batch_norm2(x)

        ## perform zero padding based on the size of the filters
        inp_filts, x_filts = inputs.shape[3], x.shape[3]
        if(inp_filts > x_filts):
            x = tf.pad(x, tf.constant([[0, 0], [0, 0], [0, 0], [0, inp_filts - x_filts]]), 'CONSTANT', constant_values = 0)
        elif(x_filts > inp_filts):
            inputs = tf.pad(inputs, tf.constant([[0, 0], [0, 0], [0, 0], [0, x_filts - inp_filts]]), 'CONSTANT', constant_values = 0)

        ## return the output of the residual block after element-wise addition
        return self.relu2(self.add_layer([inputs, x]))