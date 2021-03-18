from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from torch.utils import data
import os
import csv
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
torch.cuda.empty_cache()
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as pl
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
data_dir = "./data"
num_classes = 5
batch_size =32
num_epochs =100
from early import EarlyStopping
if __name__ == '__main__':
    def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience):
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
        #model.load_state_dict(torch.load('checkpoint.pt'))
        return  model, avg_train_losses, avg_valid_losses,  train_acc, valid_acc    
model = models.resnet18(pretrained=True)
#set_parameter_requires_grad(model, feature_extract)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
print(model)
feature_extract = True
sm = nn.Softmax(dim = 1)
train_transforms = transforms.Compose([transforms.Resize((300,300)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(15),
                                      transforms.ToTensor()
                                      ])

merge_data = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
train_data, valid_data = train_test_split(merge_data, test_size = 0.2, random_state= 123)
num_workers = 0
print("Number of Samples in Train: ",len(train_data))
print("Number of Samples in Valid: ",len(valid_data))
train_loader = torch.utils.data.DataLoader(train_data, batch_size,
     num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size, 
     num_workers=num_workers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patience = 20
optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
criterion = nn.CrossEntropyLoss()
model, train_loss, valid_loss, train_acc, valid_acc=train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience)
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
plt.xlabel('Epochs..........>')
plt.ylabel('Loss..........>')
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plotdataset4.png', bbox_inches='tight')

fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_acc)+1),train_acc, label='Training Accuracy')
plt.plot(range(1,len(valid_acc)+1),valid_acc,label='Validation Accuracy')
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

#find position of lowest validation loss

plt.xlabel('Epochs..........>')
plt.ylabel('Accuracy..........>')
plt.xlim(0, len(train_acc)+1) # consistent scale
#plt.grid(True)
plt.legend()
#plt.tight_layout()
plt.show()
fig.savefig('accplotdataset4.png', bbox_inches='tight')



