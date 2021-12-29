# import torch libraries
from torch.utils.data import DataLoader, Dataset, dataloader
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from IPython.display import Image
from torch import optim
from DenseNet121 import model as DenseNet121Model

def main(): 
    # set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    # instatiate the model
    model = None
    model = DenseNet121Model.to(device)

    DATA_DIRECTORY = 'datasetsforeverything/sample'

    transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                # transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])
    
    dataset = datasets.ImageFolder(DATA_DIRECTORY, transform=transform)
    print(dataset.classes)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    loss_function = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(4):
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item()))
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))

if __name__ == '__main__':
    main()