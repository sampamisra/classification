# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:05:05 2021

@author: Sampa
"""

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
import glob
import csv
from torch.utils.data import Dataset
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
test_dir=data_dir + '/test'
if __name__ == '__main__':    
      def test(model, criterion):
        model.to(device) 
        loss_epoch = 0
        accuracy_epoch = 0
        model.eval()
        pred = []
        true = []
        soft = []
        for step, (x, y) in enumerate(test_loader):
            model.zero_grad()
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)            
            # for majority voting
            softmax = torch.nn.Softmax(dim=1)
            s = softmax(outputs).cpu().detach().tolist()
            for i in range(len(s)):
                soft.append(s[i])
            predicted = outputs.argmax(1)
            preds = predicted.cpu().numpy()
            labels = y.cpu().numpy()
            preds = np.reshape(preds, (len(preds), 1))
            labels = np.reshape(labels, (len(preds), 1))
            for i in range(len(preds)):
                pred.append(preds[i][0].item())
                true.append(labels[i][0].item())            
            acc = (predicted == y).sum().item() / y.size(0)
            accuracy_epoch += acc
            loss_epoch += loss.item()
        cnf_matrix = confusion_matrix(true, pred)
        print('Confusion Matrix:\n', cnf_matrix)
        acc=accuracy_score(true, pred)
        print( 'Accuracy: %.3f' % acc )
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        accuracy_epoch = np.diag(cnf_matrix).sum().item() / len(true)        
        # Specificity or true negative rate
        #specificity = TN/(TN+FP) 
        #rint('specificity:\n', specificity)
        #report = classification_report(true, pred, target_names=['Covid', 'Healthy', 'Others'])
        #print(report)
        return loss_epoch, accuracy_epoch, (pred, true, soft)     
test_transforms = transforms.Compose([transforms.Resize((300,300)),
                    transforms.ToTensor()
                    ])
class TestDataset(Dataset) :
    def __init__(self, data_dir, transform=None) :
        self.all_data = sorted(glob.glob(os.path.join(data_dir,'*','*')))
        self.samples = sorted(glob.glob(os.path.join(data_dir,'*')))
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
            #label_name="berry"
        elif data_path.split("\\")[-2] == "bird" :
            label = 1
            #label_name="bird"
        elif data_path.split("\\")[-2] == "dog" :
            label = 2
            #label_name="dog"
        else :
            raise Exception('invalid path') 
        return img, label
    def __len__(self) : 
        length = len(self.all_data)
        return length 

    def getclassenumber(self) :
        return len(self.classes)
test_data=TestDataset(data_dir + "/Test", transform=test_transforms)
#print(len(dataset))
#print(dataset.getclassenumber())
num_classes=len(test_data.classes)
print("class number", num_classes)
batch_size =32
num_epochs =100
num_workers = 0
print("Number of Samples in Test: ",len(test_data))
test_loader = torch.utils.data.DataLoader(test_data, batch_size,
     num_workers=num_workers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
#model = models.resnet18()
#model.load_state_dict(torch.load('./resnet18-5c106cde.pth'))
#set_parameter_requires_grad(model, feature_extract)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
print(model)
feature_extract = True
sm = nn.Softmax(dim = 1)
criterion = nn.CrossEntropyLoss()    
model.load_state_dict(torch.load('checkpoint.pt'))
loss_epoch, accuracy_epoch, result = test(model, criterion)
preds, true, soft = result
images_path = test_loader.dataset.all_data
#print(images_path)
#label_name=test_loader.dataset.classes
#print(label_name)
# # images_path -> [ [images path, label] * 835 ]

with open(f"majority_Vgg16.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerow(["file", "prob_0", "prob_1", "prob_2", "predicted class", "Actual class" ])
    for i in range(len(preds)):
        f = images_path[i]
        #print(f)
        prob_0 = round(soft[i][0], 6)
        prob_1 = round(soft[i][1], 6)
        prob_2 = round(soft[i][2], 6)
        pred = preds[i]
        label = true[i]
        #label=label_name[i]
        wr.writerow([f, prob_0, prob_1, prob_2, pred, label])
