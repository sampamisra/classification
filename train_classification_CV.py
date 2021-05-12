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
from sklearn.model_selection import KFold
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
data_dir = "./data"
num_classes = 5
batch_size =32
num_epochs =100
from early import EarlyStopping
if __name__ == '__main__':
    def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience, i ):
        model.to(device) 
        epochs = num_epochs
        valid_loss_min = np.Inf
        train_losses = []
    # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = [] 
        train_acc, valid_acc =[],[]
        steps=0
        #valid_acc =[]
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
                # Move input and label tensors to the default device
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
        # Calculate accuracy
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
        
        # calculate average losses
            
            valid_accuracy = accuracy/len(valid_loader)          
            
            # print training/validation statistics 
            print(f"Epoch {epoch+1}/{epochs}.. ")
            #print('train Loss: {:.3f}'.format(epoch, loss.item()), "Training Accuracy: %d %%" % (train_accuracy))
            #print('Training Accuracy: {:.6f}'.format(
            #    train_accuracy))
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
        # model.load_state_dict(torch.load('checkpoint.pt'))
        # plt.title("Accuracy vs. Number of Training Epochs")
        # plt.xlabel("Training Epochs")
        # plt.ylabel("Accuracy")      
        # plt.plot(train_acc, label='Training acc')
        # plt.plot(valid_acc, label='Validation acc')
        # plt.legend(frameon=False)
        # plt.show()
        
        #model.load_state_dict(torch.load('checkpoint_{0}.pt'.format(i)))
        torch.save(model.state_dict(), "./model/checkpoint_Resnet18_{0}.pt".format(i))
        print("model saved")
        
        return  model, avg_train_losses, avg_valid_losses,  train_acc, valid_acc
# model = models.resnet18(pretrained=True)
# #set_parameter_requires_grad(model, feature_extract)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, num_classes)
model = models.resnet18(pretrained=True)
count=0
for child in model.children():
    count+=1
    if count <4:
        for param in child.parameters():
            param.requires_grad = False
            print(count)
            num_ftrs = model.fc.in_features
            model.fc =nn.Linear(num_ftrs, num_classes)
print(model)
feature_extract = True
sm = nn.Softmax(dim = 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params_to_update = model.parameters()
patience = 3
optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
criterion = nn.CrossEntropyLoss()
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

# Setup the loss fxn
#weight = torch.tensor([0.11, 0.04, 1])
#class_weights = torch.FloatTensor(weight).cuda()
#criterion = nn.CrossEntropyLoss(weight=class_weights) 
train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                                      transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(15),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])

# validation_transforms = transforms.Compose([transforms.Resize((224,224)),
#                                             transforms.ToTensor(),
#                                             transforms.Normalize([0.485, 0.456, 0.406], 
#                                                                  [0.229, 0.224, 0.225])])
merge_data = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
fold_counts= 5
kfold = KFold(n_splits=fold_counts, random_state=777, shuffle=True)
num_workers = 0
#
#--------------------------------------------------------------
for i, (train_index, validate_index) in enumerate(kfold.split(merge_data)):
    #print("train index:", train_index, "validate index:", validate_index)
    train = torch.utils.data.Subset(merge_data, train_index)
    validation = torch.utils.data.Subset(merge_data, validate_index)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    print("Number of Samples in Train: ",len(train))
    print("Number of Samples in Valid: ",len(validation))
    model, train_loss, valid_loss, train_acc, valid_acc=train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, patience, i)
