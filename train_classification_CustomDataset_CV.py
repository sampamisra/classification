# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:15:44 2021

@author: Sampa
"""

from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.utils import data
import glob
import os
from sklearn.model_selection import KFold, GroupKFold
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
torch.cuda.empty_cache()
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as pl
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
data_dir = "./Data"

from early import EarlyStopping
if __name__ == '__main__':
    def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience, i):
        model.to(device) 
        epochs = num_epochs
        valid_loss_min = np.Inf
        train_losses = []
        valid_losses = []
        avg_train_losses = []
        avg_valid_losses = [] 
        train_acc, valid_acc =[],[]
        steps=0
        best_acc = 0.0
        import time
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        for epoch in range(epochs):    
            start = time.time()            
            #scheduler.step()
            model.train()         
            total_train = 0
            correct_train = 0          
            for inputs, labels in train_loader:
                steps+=1
                inputs, labels = inputs.to(device), labels.to(device)                
                optimizer.zero_grad()                
                logps = model(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                _, predicted = torch.max(logps.data, 1)
                total_train += labels.nelement()
                correct_train += predicted.eq(labels.data).sum().item()
                train_accuracy = correct_train / total_train
                model.eval()               
            with torch.no_grad():
                accuracy = 0
                for inputs, labels in valid_loader:                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    loss = criterion(logps, labels)
                    valid_losses.append(loss.item())
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()             
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            valid_acc.append(accuracy/len(valid_loader)) 
            train_acc.append(train_accuracy)            
            valid_accuracy = accuracy/len(valid_loader) 
            print(f"Epoch {epoch+1}/{epochs}.. ")
            print('Training Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(
                train_loss, valid_loss, train_accuracy*100, valid_accuracy*100))
            train_losses = []
            valid_losses = []        
            if valid_accuracy > best_acc:
                best_acc = valid_accuracy
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break           
        print('Best val Acc: {:4f}'.format(best_acc*100))  
        torch.save(model.state_dict(), "./model/checkpoint_Resnet18_{0}.pt".format(i))
        #model.load_state_dict(torch.load('checkpoint.pt'))
        return  model, avg_train_losses, avg_valid_losses,  train_acc, valid_acc   
class TrainDataset(Dataset) :
    def __init__(self, data_dir, transform=None) :
        self.all_data = sorted(glob.glob(os.path.join(data_dir,'*','*')))
        self.transform = transform
        
        # for getclasses function
        self.classes = set()
        for data_path in self.all_data :
            self.classes.add(data_path.split("\\")[-2])

    def __getitem__(self, index) :
        data_path = self.all_data[index]
        img = Image.open(data_path)
        img = img.convert("RGB")
        if self.transform is not None :
            img = self.transform(img)

        if data_path.split("\\")[-2] == "berry" :
            label = 0
        elif data_path.split("\\")[-2] == "bird" :
            label = 1
        elif data_path.split("\\")[-2] == "dog" :
            label = 2
        else :
            raise Exception('invalid path') 
        return img, label
    def __len__(self) : 
        length = len(self.all_data)
        return length 

    def getclassenumber(self) :
        return len(self.classes)
def initialize_model(num_classes):
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model    

train_transforms = transforms.Compose([transforms.Resize((300,300)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(15),
                                      transforms.ToTensor()
                                      ])
merge_data=TrainDataset(data_dir + "/Train", transform=train_transforms)
#print(len(dataset))
#print(dataset.getclassenumber())
num_classes=len(merge_data.classes)
print("class number", num_classes)
batch_size =32
num_epochs =100
fold_counts= 5
kfold = KFold(n_splits=fold_counts, random_state=777, shuffle=True)
num_workers = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patience = 20
 
#model = models.resnet18()
#model.load_state_dict(torch.load('./resnet18-5c106cde.pth'))
#set_parameter_requires_grad(model, feature_extract)
feature_extract = True
sm = nn.Softmax(dim = 1)

criterion = nn.CrossEntropyLoss()
for i, (train_index, validate_index) in enumerate(kfold.split(merge_data)):
    #print("train index:", train_index, "validate index:", validate_index)
    train = torch.utils.data.Subset(merge_data, train_index)
    validation = torch.utils.data.Subset(merge_data, validate_index)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    model = initialize_model(num_classes)
    optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
    print("Number of Samples in Train: ",len(train))
    print("Number of Samples in Valid: ",len(validation))
    model, train_loss, valid_loss, train_acc, valid_acc=train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience, i)
