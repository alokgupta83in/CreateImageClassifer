# Imports here
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets, transforms, models

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from collections import OrderedDict
import time
import random, os

from PIL import Image

import json
from utility import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default=3)
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint1.pth")
    return parser.parse_args()

def train_model(model, epochs, gpu,criterion, optimizer, traindataloaders, validatedataloaders):
                
    device = torch.device("cuda" if gpu == 'gpu' else "cpu")
    steps = 0
    
    print_every = 5

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in traindataloaders:
            steps += 1
            #move input and label tensors to available device(Cuda).
            if gpu == 'gpu':
                model.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda') # use cuda
            else:
                model.cpu()

            # Set the gradient to 0 so that it will not retian previous gradients.
            optimizer.zero_grad()

            logps = model.forward(inputs)

            training_loss = criterion(logps, labels)

            training_loss.backward()

            optimizer.step()

            running_loss += training_loss.item()

            if steps % print_every == 0:
                testing_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs2, labels2 in validatedataloaders:
                        if gpu == 'gpu':
                            inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda') # use cuda
                            model.to('cuda:0') # use cuda
                        else:
                            pass

                        logps = model.forward(inputs2)

                        batch_loss = criterion(logps, labels2)

                        testing_loss += batch_loss.item()

                        # Calculate accuracy
                        actual_ps = torch.exp(logps)

                        top_p, top_class = actual_ps.topk(1, dim=1)

                        equals = top_class == labels2.view(*top_class.shape)

                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item() 

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {testing_loss/len(validatedataloaders):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validatedataloaders):.3f}")

                running_loss = 0
                model.train()

def train_save_model():
    
    args = parse_args()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
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

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    validate_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    traindataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testdataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
    validatedataloaders = torch.utils.data.DataLoader(validate_datasets, batch_size=64, shuffle=True)
    
    model = getattr(models, args.arch)(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    if args.arch == "densenet121":
        classifier = nn.Sequential(nn.Linear(1024, 500),
                                 nn.Dropout(0.6),
                                 nn.ReLU(),
                                 
                                 nn.Linear(500, 102),
                                 nn.LogSoftmax(dim=1))
    elif args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(nn.Linear(feature_num, 1024),
                                 nn.Dropout(0.6),
                                 nn.ReLU(),
                                 
                                 nn.Linear(1024, 102),
                                 nn.LogSoftmax(dim=1))
        
       
    criterion = nn.NLLLoss()
    model.classifier = classifier

    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))

    epochs = int(args.epochs)
    
    class_index = train_datasets.class_to_idx
    
    gpu = args.gpu
    
    train_model(model, epochs, gpu,criterion, optimizer, traindataloaders, validatedataloaders)
    
    model.class_to_idx = class_index
    
    path = args.save_dir # get the new save location 
    
    save_checkpoint(path, model, optimizer, args, classifier)
    
if __name__ == "__main__":
    train_save_model()