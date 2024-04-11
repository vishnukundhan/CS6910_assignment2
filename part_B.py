import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler

# Resizing the data as there are different image resolutions so a simple and suitable resolution is used .
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

#Splitting the data into two parts of validation and train ,20% is done for validation
trainDataset = datasets.ImageFolder("/kaggle/input/inaturalist12k/Data/inaturalist_12K/train/", transform=transform)
trainIndices, valIndices = train_test_split(list(range(len(trainDataset))), test_size=0.2, random_state=42)
trainSampler, valSampler = SubsetRandomSampler(trainIndices), SubsetRandomSampler(valIndices)

#Train function to train the model of 5 layers CNN
def trainNetwork():
    trainLoader = DataLoader(trainDataset, batch_size=32, sampler=trainSampler)
    valLoader = DataLoader(trainDataset, batch_size=32, sampler=valSampler)

    model = models.resnet50(pretrained=True)
    n_ftrs = model.fc.in_features
    #Our model has 10 layered output
    model.fc = torch.nn.Linear(n_ftrs, 10)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    lossCriteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for i in range(10):
        tempLoss = 0.0
        true = 0
        total = 0
        for inputs, labels in trainLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lossCriteria(outputs, labels)
            loss.backward()
            optimizer.step()
            tempLoss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            true += predicted.eq(labels).sum().item()

        trainLoss = tempLoss/len(trainLoader)
        trainAccuracy = 100*(true/total)
        valLoss, valAccuracy = validateNetwork(valLoader, model)

        print(f"epoch {i+1}/10: train loss: {trainLoss}, train accuracy: {trainAccuracy}, val loss: {valLoss}, val accuracy: {valAccuracy}")


def validateNetwork(valLoader,model):
    model.eval()
    valLoss = 0.0
    true = 0
    total = 0
    lossCriteria = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for inputs, labels in valLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = lossCriteria(outputs, labels)
            valLoss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            true += predicted.eq(labels).sum().item()

    valLoss /= len(valLoader)
    valAccuracy = 100*(true/total)
    return valLoss, valAccuracy

trainNetwork()