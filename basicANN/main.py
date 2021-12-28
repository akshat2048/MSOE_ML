# import torch libraries
from torch.utils.data import DataLoader, Dataset, dataloader
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from IPython.display import Image
from torch import optim

def main(): 
    # set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    # instatiate the model
    model = Net()
    model = model.to(device)

    DATA_DIRECTORY = 'datasetsforeverything/sample'

    transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])
    
    dataset = datasets.ImageFolder(DATA_DIRECTORY, transform=transform)
    print(dataset.classes)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    loss_function = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(1):
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


class Net(nn.Module):

    def __init__(self) -> None:
        super(Net, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(224 * 224 * 1, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x
    
class BIMCVSampleDataset(Dataset):
    def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image_name = os.listdir(self.data_dir)[idx]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path)
        label = image_name.split('_')[0]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    main()