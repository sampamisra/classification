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
     def test(model, criterion):
        running_corrects = 0
        running_loss=0
            #test_loss = 0.
        pred = []
        true = []
        output =[]
        pred_wrong = []
        true_wrong = []
        image = []
        for j, (inputs, labels) in enumerate(test_loader):
#            inputs = inputs.to(device)
#            labels = labels.to(device)
            model.eval()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
                #test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            
            outputs = sm(outputs)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            preds = np.reshape(preds,(len(preds),1))
            labels = np.reshape(labels,(len(preds),1))
            inputs = inputs.cpu().numpy()
                
            for i in range(len(preds)):
                pred.append(preds[i])
                true.append(labels[i])
                if(preds[i]!=labels[i]):
                    pred_wrong.append(preds[i])
                    true_wrong.append(labels[i])
                    image.append(inputs[i])      
        mat_confusion=confusion_matrix(true, pred)
            #f1_score = f1_score(true,pred)
        print('Confusion Matrix:\n',mat_confusion)
        matrix = confusion_matrix(true, pred)
        acc=accuracy_score(true, pred)
        print( 'Accuracy: %.3f' % acc )
        #matrix = matrix.astype('float')
#cm_norm = matrix / matrix.sum(axis=1)[:, np.newaxis]
        #print(matrix)
#class_acc = np.array(cm_norm.diagonal())
        # class_acc = [matrix[i,i]/np.sum(matrix[i,:]) if np.sum(matrix[i,:]) else 0 for i in range(len(matrix))]
        # print('Sens Normal: {0:.3f}, Pneumonia: {1:.3f}, COVID-19: {2:.3f}'.format(class_acc[0],
        #                                                                    class_acc[1],
        #                                                                    class_acc[2]))
                                                                           
        # ppvs = [matrix[i,i]/np.sum(matrix[:,i]) if np.sum(matrix[:,i]) else 0 for i in range(len(matrix))]
        # print('PPV Normal: {0:.3f}, Pneumonia {1:.3f}, COVID-19: {2:.3f}'.format(ppvs[0],
        #                                                                  ppvs[1],
        #                                                                  ppvs[2]))
       
        #auc = roc_auc_score(np.asarray(true).ravel(), out[:,1])
        #print('ROC AUC: %f' % auc)
        # confusion matrix
       
        #fpr, tpr, thresholds = roc_curve(np.asarray(true).ravel(), out[:,1])
    
        # plt.figure()
        # plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc)
        # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        # #plt.xlim([0.0, 1.0])
        # #plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic')
        # plt.legend(loc="lower right")
        # plt.show()

        return 
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
test(model, criterion) 

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

