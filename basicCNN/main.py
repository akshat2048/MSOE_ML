# import main libraries
import pandas as pd
import numpy as np

# import torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

def analyzeImageData():
    # read in the data
    data = pd.read_csv('basicCNN/chest_x_ray_images_labels_sample.csv')
    
    # get number of unique patient IDs
    uniquePatientIDs = data['PatientID'].unique()
    print('Number of unique patient IDs:', len(uniquePatientIDs))

    # get number of rows
    numRows = data.shape[0]
    print('Number of rows:', numRows)

    # get value counts for the projection series
    projectionSeries = data['Projection']
    projectionSeriesValueCounts = projectionSeries.value_counts()
    print('Value counts for the projection series:')
    print(projectionSeriesValueCounts)

    # get the shape of the data
    # print(data.shape)

    # get the number of unique classes
    # print(data['label'].unique())

    # get the number of unique classes
    # print(len(data['label'].unique()))

    # get the number of images in each class
    # print(data['label'].value_counts())

    # get the number of images in each class
    # print(data['label'].value_counts().shape)

    # get the number of images in each class
    # print(data['label'].value_counts().sum())

    # get the number of images in each class
    # print(data['label'].value_counts().sum() / data['label'].value_counts().shape[0])

    # get the number of images in each class
    # print(data['label'].value_counts().sum() / data['label'].value_counts().shape[0])

    # get the number of images in each class
    # print(data['label'].value_counts().sum() / data['label'].value_counts().shape[0])

    # get the number of images in each class
    # print(data['label'].value_counts().sum() / data['label'].value_counts().shape[0])

    # get the number of images in each class
    # print(data['label'].value_counts().sum() / data['label'].value_counts().shape[0])

    # get the number of images in each class
    # print(data['label'].value_counts().sum() / data['label'].value_counts().shape[0])

    # get the number of images in each class
    # print(data['label'].value_counts().sum() / data['label'].value_counts().shape[0])

    # get the number of images in each class
    # print(data['label'].

def main():
    # net = Net()
    # print(net)
    # params = list(net.parameters())
    # print("Number of parameters:", len(params))
    # print(params[0].size())
    # input = torch.randn(1, 1, 32, 32)
    # out = net(input)
    # print(out)
    # net.zero_grad()
    # out.backward(torch.randn(1, 10))    
    net = DenseNet121(14)
    net.load_state_dict(torch.load('basicCNN/model.pth.tar', map_location='cpu'))
    print(net)

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = tv.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

if __name__ == '__main__':
    main()