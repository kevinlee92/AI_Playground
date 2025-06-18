# TODO: Do validation on the test set

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
import matplotlib as plt
import numpy as np

from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Args for Flower Classifier')

parser.add_argument('checkpoint', type=str, help='Path and Name of the checkpoint file')
parser.add_argument('--data_dir', action='store', default='flower_data')
parser.add_argument('--top_k', action='store', default=3, type=int)
parser.add_argument('--category_names', action='store', default='cat_to_name.json', type=str)
parser.add_argument('--hidden_units', action='store', default=128, type=int) #if different than default in checkpoint, this script will fail
parser.add_argument('--gpu', action='store', default=True, type=bool)

args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)

validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg19(pretrained=True)
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Rebuild the classifier
    model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(args.hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checkpoint['state_dict'])
    print(model)
    
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move input and label tensors to the default device
valid_loss = 0
accuracy = 0
steps = 0
criterion = nn.NLLLoss()
model = load_checkpoint(args.checkpoint)
model.to(device)
model.eval()
with torch.no_grad():
    for inputs, labels in validloader:
        steps += 1
        print("Currently on Validation step %d" % steps)
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        valid_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        # print(labels, "LABELS")
        # print(top_class,"TOP CLASS")
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Validation loss: {valid_loss/len(validloader):.3f}.. "
      f"Validation accuracy: {accuracy/len(validloader):.3f}")