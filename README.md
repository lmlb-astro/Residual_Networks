A set of residual network models for computer vision purposes. It build on residual blocks with 2 convolutional layers each (i.e. ResBlock2())
The number in the title of the model defines the number of layers in the network. These layers include:
- 1 convolutional layer to start
- A number of convolutional layers defined in the residual blocks that are being called
- 1 dense layer at the end for the output
