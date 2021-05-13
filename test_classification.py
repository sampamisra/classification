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
test_dir=data_dir + '/test'
num_classes = 5
batch_size =32
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('checkpoint.pt'))
if __name__ == '__main__':    
    def test(model, criterion, optimizer):
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
        report = classification_report(true, pred, target_names=['A', 'B', 'C', 'D', 'E'])
        print(report)
        return loss_epoch, accuracy_epoch, (pred, true, soft)     
feature_extract = True
sm = nn.Softmax(dim = 1)
test_transforms = transforms.Compose([transforms.Resize((300,300)),
                    transforms.ToTensor()
                    ])
test_data= datasets.ImageFolder(test_dir,transform=test_transforms)
num_workers = 0
print("Number of Samples in Test ",len(test_data))
test_loader = torch.utils.data.DataLoader(test_data, batch_size, 
     num_workers=num_workers, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer= optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
criterion = nn.CrossEntropyLoss()
loss_epoch, accuracy_epoch, result = test(model, criterion, optimizer)

# preds, true, soft = result
# images_path = test_loader.dataset.samples
# # images_path -> [ [images path, label] * 835 ]

# with open(f"majority.csv", "w") as f:
#     wr = csv.writer(f)
#     wr.writerow(["file", "prob_0", "prob_1", "prob_2", "prob_3", "prob_4",  "pred", "label"])
#     for i in range(len(preds)):
#         f = os.path.basename(images_path[i][0])
#         prob_0 = round(soft[i][0], 6)
#         prob_1 = round(soft[i][1], 6)
#         prob_2 = round(soft[i][2], 6)
#         prob_3 = round(soft[i][3], 6)
#         prob_4 = round(soft[i][4], 6)
#         pred = preds[i]
#         label = true[i]
#         wr.writerow([f, prob_0, prob_1, prob_2, prob_3, prob_4, pred, label])

