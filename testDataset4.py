
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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
torch.cuda.empty_cache()
import pandas as pd
from torch.optim.lr_scheduler import StepLR

data_dir = "./DatasetD_1"
test_dir=data_dir + '/test'
model_name = "resnet18"
# Models to choose from [resnet18, resnet50, alexnet, vgg, squeezenet, densenet, inception]

#test_dir='E:/Sampa_Opticho/WorkWithRavi/Data_RaviNew11.06/Data3/BW/Test'
# Models to choose from [resnet18, resnet50, alexnet, vgg, squeezenet, densenet, inception]


# Number of classes in the dataset
num_classes = 3

# Batch size for training (change depending on how much memory you have)
batch_size = 32
class_names = ['Normal', 'Pneumonia', 'Covid']
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

feature_extract = True
sm = nn.Softmax(dim = 1)
test_transforms = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225])
                    #transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
                    ])

test_data= datasets.ImageFolder(test_dir,transform=test_transforms)

#targets = datasets.ImageFolder.targets

num_workers = 0

print("Number of Samples in Test ",len(test_data))

test_loader = torch.utils.data.DataLoader(test_data, batch_size, 
     num_workers=num_workers, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Setup the loss fxn
weight = torch.tensor([0.11, 0.04, 1])
#class_weights = torch.FloatTensor(weight)
class_weights = torch.FloatTensor(weight)
criterion = nn.CrossEntropyLoss(weight=class_weights) 
if __name__ == '__main__':
    # def imshow(inp, title=None):
    #     inp = inp.numpy().transpose((1, 2, 0))
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     inp = std * inp + mean
    #     inp = np.clip(inp, 0, 1)
    #     plt.imshow(inp)
    #     if title is not None:
    #         plt.title(title)
    #     plt.pause(0.1)  # pause a bit so that plots are updated
    # inputs, classes = next(iter(test_loader))
    # out = torchvision.utils.make_grid(inputs)
    # imshow(out, title=[class_names[x] for x in classes])
    def test(model, criterion):
        #model.to(device) 
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
            #inputs = inputs.to(device)
            #labels = labels.to(device)
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
                output.append(outputs[i])
                if(preds[i]!=labels[i]):
                    pred_wrong.append(preds[i])
                    true_wrong.append(labels[i])
                    image.append(inputs[i])
            
    
        out=[]
        for k in range(len(true)):
          abc=output[k].cpu()
          xyz=abc.detach().squeeze(-1).numpy()
          out.append(xyz)
        out=np.asarray(out)        
        mat_confusion=confusion_matrix(true, pred)
            #f1_score = f1_score(true,pred)
        print('Confusion Matrix:\n',mat_confusion)
        matrix = confusion_matrix(true, pred)
        acc=accuracy_score(true, pred)
        print( 'Accuracy: %.3f' % acc )
       # matrix = matrix.astype('float')
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
        target_names = ['Normal', 'Pneumonia', 'Covid']
        print(classification_report(true, pred, target_names=target_names))
        kappa = cohen_kappa_score(true, pred)
        print('Cohens kappa: %f' % kappa)
        mc= matthews_corrcoef(true, pred)
        print('Correlation coeff: %f' % mc)
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

        return out

       
        #print('Precision: {}, Recall: {} '.format(precision*100, recall*100))
     
    feature_extract = True
    sm = nn.Softmax(dim = 1)
    #model= models.squeezenet1_1(pretrained=True)
    def set_parameter_requires_grad(model, feature_extracting):
            if feature_extracting:
                for param in model.parameters():
                    param.requires_grad = True    
    def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
        model = None
        input_size = 0

        if model_name == "resnet18_1":
            """ Resnet18
            """
            model = models.resnet18(pretrained=True)
            #print(model)
            count=0
            # for child in model.children():
            #     count+=1

            # print('No. of layers')
            # print(count)
            # count = 0
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
            # model.fc = nn.Sequential(
            #     nn.Dropout(0.5),
            #     nn.Linear(num_ftrs, num_classes))
            # input_size = 224    

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
    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()
        

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {} truth: {}'.format(class_names[preds[j]], class_names[labels[j]]))
                    img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                    img = std * img + mean
                    ax.imshow(img)

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)    

# Initialize the model for this run
model = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
#model.load_state_dict(torch.load('Checkpoint_final\checkpoint_D4_5.pt'))
model.load_state_dict(torch.load('checkpoint_D4_5.pt'))


#targets = datasets.ImageFolder.targets


#criterion = nn.CrossEntropyLoss()        
torch.cuda.empty_cache()
out=test(model, criterion) 
#visualize_model(model, num_images=6)
#print(out)



