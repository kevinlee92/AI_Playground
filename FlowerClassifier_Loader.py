

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
import matplotlib as plt
import numpy as np

from PIL import Image

import argparse


parser = argparse.ArgumentParser(description='Args for Flower Classifier')

parser.add_argument('image_path', type=str, help='Path to the input image')
parser.add_argument('checkpoint', type=str, help='Path and Name of the checkpoint file')
parser.add_argument('--top_k', action='store', default=3, type=int)
parser.add_argument('--category_names', action='store', default='cat_to_name.json', type=str)
parser.add_argument('--hidden_units', action='store', default=128, type=int) #if different than default in checkpoint, this script will fail
parser.add_argument('--gpu', action='store', default=True, type=bool)

args = parser.parse_args()

print(f"Image path: {args.image_path}")
print(f"Checkpoint path: {args.checkpoint}")
print(f"Category names path: {args.category_names}")

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

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open the image
    image = Image.open(image_path)
    
    # Resize the image where the shortest side is 256 pixels, keeping the aspect ratio
    if image.size[0] < image.size[1]:
        image.thumbnail((256, 256 * image.size[1] // image.size[0]))
    else:
        image.thumbnail((256 * image.size[0] // image.size[1], 256))
    
    # Crop out the center 224x224 portion of the image
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    
    # Convert the image to a numpy array
    np_image = np.array(image) / 255.0
    print(np_image)
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions to match PyTorch's expectations
    np_image = np_image.transpose((2, 0, 1))
    print(np_image)
    
    return np_image

# Point to a photo and run it
image_path = args.image_path #"C:/Users/leeka3/Documents/Udacity/AI & Python/flower_data/test/1/image_06743.jpg"
processed_image = process_image(image_path)

from matplotlib import pyplot as plt
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#imshow(processed_image)

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
def predict(np_image, model, topk=args.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()

    
    image = torch.from_numpy(np_image).type(torch.FloatTensor)
    image = image.unsqueeze_(0) #The model expects the input shape to be [batch_size, 3, 224, 224]

    image = image.to(device)

    with torch.no_grad():
        preds = model.forward(image)
        probs = torch.exp(preds)
        top_k, top_class = probs.topk(topk, dim=1)
    print(top_k) # probabilites in tensor format
    print(top_class) # associated classes predicted (+1?)
    return top_k, top_class

checkpoint = args.checkpoint

#predict(processed_image, load_checkpoint(checkpoint))

imshow(processed_image)

import json
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

top_k, top_class = predict(processed_image, load_checkpoint(checkpoint))

top_k = top_k.cpu().numpy().tolist()[0]
top_class = top_class.cpu().numpy().tolist()[0]

print(top_k)
print(top_class)

labels = []
for num in top_class:
    num = num+1
    if str(num) in cat_to_name:
        label = cat_to_name[str(num)]
        labels.append(label)
        print(label)
        
def bar_graph(categories, values, title="Flower Probabilities", xlabel="Flowers", ylabel="Probability"):

    fig, ax = plt.subplots()
    ax.barh(categories, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

bar_graph(labels, top_k, title="Flower Probabilities", xlabel="Flowers", ylabel="Probability")
