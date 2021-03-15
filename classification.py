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
model_name = "resnet50"
num_classes = 3
batch_size =32
num_epochs =500
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
        model.load_state_dict(torch.load('checkpoint.pt'))
        return  model, avg_train_losses, avg_valid_losses,  train_acc, valid_acc    
   
    def test(model, criterion):
        model.to(device) 
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
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.eval()
            outputs = model(inputs)
            loss = criterion(outputs, labels)            
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
                output.append(outputs[i])
                if(preds[i]!=labels[i]):
                    pred_wrong.append(preds[i])
                    true_wrong.append(labels[i])
                    image.append(inputs[i])       
        mat_confusion=confusion_matrix(true, pred)
        print('Confusion Matrix:\n',mat_confusion)
        acc=accuracy_score(true, pred)
        print( 'Accuracy: %.3f' % acc )
        target_names = ['Normal', 'Pneumonia', 'Covid']
        print(classification_report(true, pred, target_names=target_names))
        kappa = cohen_kappa_score(true, pred)
        print('Cohens kappa: %f' % kappa)
        mc= matthews_corrcoef(true, pred)
        print('Correlation coeff: %f' % mc)
        return      
    feature_extract = True
    sm = nn.Softmax(dim = 1)
    def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = True    
    def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        model = None
        input_size = 0
        if model_name == "resnet18_1":
            """ Resnet18
            """
            model = models.resnet18(pretrained=True)
            count=0
            for child in model.children():
                count+=1
                if count <4:
                    for param in child.parameters():
                        param.requires_grad = False
                    print(count)    
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, num_classes))
            input_size = 224
            #print(model)
        elif model_name == "resnet18":
            """ Resnet18
            """
            model = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224    

        elif model_name == "alexnet1":
            """ Alexnet
            """
            model= models.alexnet(pretrained=True)
            count=0
            for child in model.children():
                count+=1
            print('No. of layers')
            print(count)
            count = 0  
            for child in model.children():
                if count <2:
                    for param in child.parameters():
                        param.requires_grad = False  
                    count+=1        
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224
        elif model_name == "alexnet":
            """ Alexnet
            """
            model= models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224
        elif model_name == "resnet50":
            """ Resnet50
            """
            model = models.resnet50(pretrained=use_pretrained)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
        elif model_name == "vgg1":
            """ VGG19_bn
            """
            model = models.vgg19_bn(pretrained= True)
            count=0
            for child in model.children():
                count+=1
            print('No. of layers')
            print(count)
            count = 0  
            for child in model.children():
                count+=1
                if count <2:
                    for param in child.parameters():
                        param.requires_grad = False  
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224  

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model = models.vgg19_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model= models.squeezenet1_1(pretrained=use_pretrained)
            #set_parameter_requires_grad(model, feature_extract)
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model.num_classes = num_classes
            input_size = 224
        elif model_name == "squeezenet1":
            """ squeezenet
            """
            model =  models.squeezenet1_1(pretrained= True)
            count=0
            for child in model.children():
                count+=1
            print('No. of layers')
            print(count)
            count = 0  
            for child in model.children():
                count+=1
                if count <2:
                    for param in child.parameters():
                        param.requires_grad = False  
            model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model.num_classes = num_classes
            input_size = 224    

        elif model_name == "densenet":
            """ Densenet
            """
            model= models.densenet121(pretrained=use_pretrained)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "densenet1":
            """ densenet
            """
            model= models.densenet121(pretrained= True)
            count=0
            for child in model.children():
                count+=1
            print('No. of layers')
            print(count)
            count = 0  
            for child in model.children():
                count+=1
                if count <2:
                    for param in child.parameters():
                        param.requires_grad = False  
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224      

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model = models.inception_v3(pretrained=use_pretrained)
            set_parameter_requires_grad(model, feature_extract)
            # Handle the auxilary net
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model

# Initialize the model for this run
    model = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    print(model)                        

train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                                      transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(15),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])


test_transforms = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225])
                    #transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
                    ])
merge_data = datasets.ImageFolder(data_dir + "/train", transform=train_transforms)
train_data, valid_data = train_test_split(merge_data, test_size = 0.2, random_state= 123)
test_data= datasets.ImageFolder(test_dir,transform=test_transforms)
num_workers = 0
print("Number of Samples in Train: ",len(train_data))
print("Number of Samples in Valid: ",len(valid_data))
print("Number of Samples in Test ",len(test_data))
train_loader = torch.utils.data.DataLoader(train_data, batch_size,
     num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size, 
     num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size, 
     num_workers=num_workers, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params_to_update = model.parameters()
#print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            #print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
patience = 20
optimizer= optim.SGD(params_to_update, lr=0.001, momentum=0.8)
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
test(model, criterion) 


