## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        # input 1x224x224
        self.conv1 = nn.Conv2d(1, 32, 5)  #32x221x221
        self.pool1 = nn.MaxPool2d(4, 4)   #32x55x55
        self.drop1 = nn.Dropout(0.1)
       
        self.conv2 = nn.Conv2d(32, 64, 3) #64x53x53
        self.pool2 = nn.MaxPool2d(4, 4)   #64x13x13
        self.drop2 = nn.Dropout(0.2)

        self.lin1 = nn.Linear(64*13*13,500)
        self.drop3 = nn.Dropout(0.3)
        self.lin2 = nn.Linear(500,68*2)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        
        x = x.view(x.size(0), -1) # flatten
        
        x = self.drop3(F.relu(self.lin1(x)))
        x = self.lin2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x