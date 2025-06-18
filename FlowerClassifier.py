# Check torch version and CUDA status if GPU is enabled.
import torch
print(torch.__version__)
print(torch.cuda.is_available()) # Should return True when GPU is enabled. 

# Imports here
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from pathlib import Path
from torchvision import datasets, transforms, models
from torch import optim
import json
import argparse

parser = argparse.ArgumentParser(description='Args for Flower Classifier')

parser.add_argument('--save_dir', action='store', default='./')
parser.add_argument('--data_dir', action='store', default='flower_data')
parser.add_argument('--arch', action='store', default='vgg19')
parser.add_argument('--learning_rate', action='store', default=0.001, type=float)
parser.add_argument('--hidden_units', action='store', default=128, type=int)
parser.add_argument('--epochs', action='store', default=5, type=int)
parser.add_argument('--gpu', action='store', default=True, type=bool)

args = parser.parse_args()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# #print(cat_to_name)
# test = 4
# #print(cat_to_name.keys())
# if str(test) in cat_to_name:
#     print(cat_to_name[str(test)])

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
###########data_transforms = 
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
####image_datasets = 
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
####dataloaders = 

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

arch_to_use = args.arch
model = models.vgg19(pretrained=True) if args.arch == 'vgg19' else models.arch_to_use(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
## In = 25088 (224x224 /2)
## Out = 102 (number of choices available)
model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(args.hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

model.to(device)

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        print("Currently on step %d" % steps)
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()

## View it
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())

## Save it
checkpoint = {'input_size': 25085,
              'output_size': 102,
              'state_dict': model.state_dict()}
torch.save(checkpoint, args.save_dir + 'checkpoint2.pth')

