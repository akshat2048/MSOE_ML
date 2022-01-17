# import torch libraries
from torch.utils.data import DataLoader, Dataset, dataloader
import torch
import torch.nn as nn
from torchvision import datasets, transforms
# from IPython.display import Image
from torch import optim
from DenseNet121 import model as DenseNet121Model
from environmentsettings import settings

TRAINING_DIRECTORY = settings['TRAINING_DIRECTORY']
TESTING_DIRECTORY = settings['TESTING_DIRECTORY']
LEARNING_RATE = settings['LEARNING_RATE']
BATCH_SIZE = settings['BATCH_SIZE']
EPOCHS = settings['EPOCHS']

def main(): 
    # set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    # instatiate the model
    model = None
    model = DenseNet121Model.to(device)

    #Transformations to apply to image tensors
    transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                # transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])

    # Set up the TRAINING dataset and dataloader
    trainingSet = datasets.ImageFolder(TRAINING_DIRECTORY, transform=transform)
    trainLoader = DataLoader(trainingSet, batch_size=BATCH_SIZE, shuffle=True)

    # Set up the TESTING dataset and dataloader
    testSet = datasets.ImageFolder(TESTING_DIRECTORY, transform=transform)
    testLoader = DataLoader(testSet, batch_size=BATCH_SIZE, shuffle=True)

    # define the loss function
    loss_function = nn.NLLLoss()

    # define the optimizer with the parameters and learning rate
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #Train the model
    model.train()

    correct = 0

    predicted = []
    targets = []
    corrects = []
    labelings = []

    for e in range(EPOCHS):
        running_loss_for_this_epoch = 0
        for id, (images, labels) in enumerate(trainLoader):
            data, target = images.to(device), labels.to(device)
            

            # zero the parameter gradients
            optimizer.zero_grad()

            # calculate the output
            output = model(data)

            predicted_value = output.max(1, keepdim=True)[1]
            isCorrect = predicted_value.eq(target.view_as(predicted_value)).sum().item()
            correct += isCorrect

            predicted.append(predicted_value)
            targets.append(target)
            corrects.append(isCorrect)
            labelings.append(id)

            # calculate the loss
            loss = loss_function(output, target)

            # calculate the gradients
            loss.backward()

            # update the parameters
            optimizer.step()

            # update loss
            running_loss_for_this_epoch += loss.item()
        
        # print stats
        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2e}'.format(
            e, correct, len(trainLoader.dataset),
            100*(correct / len(trainLoader.dataset)), running_loss_for_this_epoch))
        correct = 0
        running_loss_for_this_epoch = 0

    # Test the model
    model.eval()

    running_loss_for_this_epoch = 0
    correct = 0

    print("The predicted and target values are: ")

    for (pred, target, c, label) in zip(predicted, targets, corrects, labelings):
        print(f"For this {label} image, ", "The predicted value is:",pred, ", the actual value is:", target, ". The model found this to be correct:", c)

    predicted = []
    targets = []
    corrects = []
    labelings = []

    with torch.no_grad():
        for images, labels in testLoader:
            data, target = images.to(device), labels.to(device)

            # calculate the output
            output = model(data)

            # calculate the loss
            loss = loss_function(output, target)

            # update loss
            running_loss_for_this_epoch += loss.item()
        
            # get predicted value
            predicted_value = output.max(1, keepdim=True)[1]
            isCorrect = predicted_value.eq(target.view_as(predicted_value)).sum().item()
            correct += isCorrect

            predicted.append(predicted_value)
            targets.append(target)
            corrects.append(isCorrect)
            labelings.append(id)

    # average the loss
    running_loss_for_this_epoch /= len(testLoader.dataset)

    print("The predicted and target values are: ")
    for (pred, target, c, label) in zip(predicted, targets, corrects, labelings):
        print(f"For this {label} image, ", "The predicted value is:",pred, ", the actual value is:", target, ". The model found this to be correct:", c)

    # print stats
    print('\nTest set: Average loss: {:.2e}, Accuracy: {}/{} ({:.0f}%)\n'.format(running_loss_for_this_epoch, correct, len(testLoader.dataset), 100. * (correct / len(testLoader.dataset))))
    
    

if __name__ == '__main__':
    main()