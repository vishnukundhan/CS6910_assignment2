Q1) Aim : To classify given images to a certain class and the given image can be of RGB or gray scale

Training a CNN model to classify the image dataset (iNaturalist)

The CNN mainly comprises of convolution layer with pooling layer which plays the key role in reducing the computation

Convolutional Neural Network :
Like any other deep neural network, CNN to has layer of neurons associated with an activation function but the key difference is that as the layers go by the dimension of the feature space reduces mainly because of the convolution layers and the pooling layers which use kernels ( filters ) to perform convolution on each submatrix to generate certain value to represent the whole submatrix information.
The scale as which the feature space reduces depends on the kernel size used, stride and the padding used for the convolution operation. Its more like a sliding matrix that covers whole matrix to generate a new feature instance from the old feature information.

The current CNN model is with 5 convolutional blocks each with a convolution layer, Dropout layer, Maxpool layer

The parameters can be changed for each block using the arguments and the option to enable dropout or disable it and to use batch normalization or not is also incorporated into the model.

Followed by a dense layer which is precedded by flattening of the feature space so that the feedforward layer can be accommodated afterwards and an output layer with output neurons to be no of class available (10 in current dataset)

Model validation accuracy = 35.1%

Model Testing accuracy = 46.5%

Best Model Parameters:
Activation Function: SELU
Batch Size: 32
Dropout : 0.1
Epochs : 10
Filter Customization : [32,64,128,256,512]
Kernel Dimensions: 3*3
Learning Rate : 0.0001
Optimizer : Adam
Weight Decay : 0
Batch Normalization : True
Data Augmentation : True


Q2) Fine tuning a pretrained model
Pretrained models  are ones that are already rigroously finetuned to certain parameters and are trained on extensive dataset to handle corresponding to certain problem

Consider ResNet or VGGNet models which are initially trained to classify an image into 1000 classes as the imagenet has 1000 classes, but considering the current objective of 10 class problem the model architecture must be altered.

So need to first resize the image to 224 x 224 as the pretrained model was also trained upon the image of dimension 224 x 224 incase of ResNet

The last output layer (Fully connected layer with 1000 output channel and be changed to 10 to accommodate the challenge to classify the images to required classes and also need to train this layer weights by keeping the rest intact during the training process.

It resulted in an accuracy of 76.1% over 10 Epochs

Some parameters :
Batchsize : 32
Epochs : 10
Learning rate : 0.001
Input Image : 224 x 224 x 3
Dense Layer neurons : 512
