import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as functional
from torchvision.utils import save_image


from neighbours import find_neighbours
from classifier import GaussianKernels
from loader import MultiFolderLoader
#from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.autograd import Variable

import numpy as np
import os
import subprocess
import argparse
import scipy

from sklearn.manifold import TSNE

#import auto_augment as ag
from utils import *


######################## imports Utils ############################################
#from sklearn.neighbors import (NeighborhoodComponentsAnalysis,
#KNeighborsClassifier)
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import pairwise_distances
#from scipy.spatial.distance.cdist import distance
#from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


parser = argparse.ArgumentParser(description="Train Gaussian kernel classifier using Resnet18 or 50.")
parser.add_argument("--data_dir", required=True, type=str, help="Path to data parent directory.")
parser.add_argument("--unlabels", required=True, type=str, help="Path to data parent directory.")
parser.add_argument("--test", required=True, type=str, help="Path to data parent directory.")
parser.add_argument("--load_dir", required=True, type=str, help="Models are saved to this directory.")
parser.add_argument("--num_classes", required=True, type=int, help="Number of training classes to use.")

parser.add_argument("--im_ext", default="jpg", type=str, help="Dataset image file extensions (e.g. jpg, png).")
parser.add_argument("--gpu_id", default=None, type=int, help="GPU ID. CPU is used if not supplied.")
parser.add_argument("--sigma", default=10, type=int, help="Gaussian sigma.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--max_epochs", default=50, type=int, help="Maximum training length (epochs).")
parser.add_argument("--topk", default=20, type=int, help="top k.")
parser.add_argument("--input_size", default=256, type=int, help="input size img.")

parser.add_argument("--name", default=" ", required=True, type=str, help="Dataset file name extensions (e.g. cifar10, cifar100).")
####MIXUP
parser.add_argument('--alpha', default=1, type=float,help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--scale_mixup', default=0.0001, type=float,help='scaling the mixup loss')
parser.add_argument('--beta', default=1, type=float,help='scaling the gauss loss')
#### TSNE GRAPH
parser.add_argument('--tsne_graph', default=True, type=str, help='if true save tsne imagen')
args = parser.parse_args()

"""
TensorBorder
"""
#writer = SummaryWriter(comment="-"+args.data_dir+"-"+str(args.max_epochs)+"ep-scale_mixup "+str(args.scale_mixup)+"-alpha"+str(args.alpha)+"-beta"+str(args.beta))

seed =args.name+"-EP"+str(args.max_epochs)+"-SM"+str(args.scale_mixup)+"-A"+str(args.alpha)+"-B"+str(args.beta)
print('seed==>',seed)

result_model = list()
result_model.append("SEED::  "+str(seed)+ "\n")
result_model.append("epochs::  "+str(args.max_epochs)+ "scale_mixup::  "+str(args.scale_mixup)+ "alpha::  "+str(args.alpha)+  "beta::  "+str(args.beta)+ "\n")
result_model.append("============================= \n")


"""
Configuration
"""

HLoss = HLoss()

#Data info
input_size = args.input_size   #32 #256
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]

#Resnet18 model
model = torchvision.models.resnet50(pretrained=True)

#Remove fully connected layer
modules = list(model.children())[:-1]

#--------------------------------------------#
from collections import OrderedDict

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
#--------------------------------------------#

#modules.append(nn.Flatten())
modules.append(Flatten())
model = nn.Sequential(*modules)

#kernel_weights_lr = args.learning_rate*1
num_neighbours    = 200
#eval_interval     = args.update_interval

#Set GPU ID or 'cpu'
if args.gpu_id is None:
	device = torch.device('cpu')
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
	device = torch.device('cuda:0')

"""
Set up DataLoaders
"""

#Transformations/pre-processing operations
train_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

update_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])


train_dataset  = MultiFolderLoader(args.data_dir, train_transforms, num_classes = args.num_classes, start_indx = 0, img_type = "."+args.im_ext, ret_class=True)
update_dataset = MultiFolderLoader(args.data_dir, update_transforms, num_classes = args.num_classes, start_indx = 0, img_type = "."+args.im_ext, ret_class=True)
unlabels_dataset = MultiFolderLoader(args.unlabels, train_transforms, num_classes = args.num_classes, start_indx = 0, img_type = "."+args.im_ext, ret_class=True)
test_dataset = MultiFolderLoader(args.test, train_transforms, num_classes = args.num_classes, start_indx = 0, img_type = "."+args.im_ext, ret_class=True)

#Data loaders to handle iterating over datasets
train_loader  = DataLoader(train_dataset,  batch_size=args.batch_size, shuffle=True,  num_workers=3)
update_loader = DataLoader(update_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)
unlabels_loader = DataLoader(unlabels_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3)

"""
Create Gaussian kernel classifier
"""
model = model.to(device)
model.load_state_dict(torch.load(args.load_dir + "/"+seed+"model.pt",map_location=device))
model = model.eval()

num_train = len(update_loader.dataset)
print(num_train)


with torch.no_grad():
	num_dims = model(torch.randn(1,3,input_size,input_size).to(device)).size(1)

#Create tensor to store kernel centres
centres = torch.zeros(num_train,num_dims).type(torch.FloatTensor).to(device)
print("Size of centres is {0}".format(centres.size()))

#Create tensor to store labels of centres
centre_labels = torch.LongTensor(update_dataset.get_all_labels()).to(device)

#Create Gaussian kernel classifier
kernel_classifier = GaussianKernels(args.num_classes, num_neighbours, num_train, args.sigma)
kernel_classifier = kernel_classifier.to(device)

############################ Load best state model ######################################
kernel_classifier.load_state_dict(torch.load(args.load_dir + "/"+seed+"classifier.pt"))
centres = torch.load(args.load_dir + "/"+seed+"centres.pt")
print(centres)
model = model.eval()
########################################################################################
#######10% labeled ##############

print("#XL labeled")

feature_t= []
labels_t = []
pred_t = []
#img_t = []
running_correct = 0
#list_metric_labeld = []
for i, data in enumerate(tqdm(train_loader), 0):
    inputs, labels, indices = data
    inputs  = inputs.to(device)
    labels  = labels.to(device).view(-1)
    indices = indices.to(device)
    output = model(inputs)
    dist_matrix = torch.cdist(output, centres)
    neighbours_tr = torch.argsort(dist_matrix)[:,0:num_neighbours]
    indices = np.arange(0,output.size(0))
    log_prob, prob_real = kernel_classifier( output , centres, centre_labels, neighbours_tr[indices, :] )
    pred = log_prob.argmax(dim=1, keepdim=True)
 
    #Mcscore_l = MCScore(log_prob).cpu().detach().numpy()
    #entropy_l =  HLoss(prob_real.cpu().detach().numpy())
    #prob_max_l = prob_real.max(1).values.cpu().detach().numpy()

    #for it in range(len(log_prob)):
      #list_metric_labeld.append([Mcscore_l[it],entropy_l[it],prob_max_l[it],labels[it].cpu().item(),pred[it].cpu().item()])

    feature_t.append(output.data.cpu().numpy())
    labels_t.append(labels.data.cpu().numpy())
    pred_t.append(pred.data.cpu().numpy())
    #img_t.append(inputs.data.cpu().numpy())
    
    correct = pred.eq(labels.view_as(pred)).sum().item()
    running_correct += correct/args.batch_size


#data_L = np.array(list_metric_labeld).astype(np.float)
#scaler.fit(data_L[:,0:3])
#new_data_L = scaler.transform(data_L[:,0:3]) # normaliza os rotulados 10%

#LIMIAR = np.mean(new_data_L,axis=0) # limiares definidos pela média dos 10% rotulados em cima da média das três métricas
#STD = np.std(new_data_L, axis=0)
#print("LIMIAR XL",LIMIAR)
#print("STD XL",STD)


print('####### AAC_Label = ',running_correct/len(train_loader))

result_model.append("============================= \n")
result_model.append("AAC_Label XL::  "+str(running_correct/len(train_loader))+ "\n")

if(args.tsne_graph == "True"):
  feature_l,pred_l, true_l = unmount_batch_v2(feature_t,pred_t,labels_t)
  view_tsne = TSNE(random_state=123).fit_transform(feature_l)
  plt.scatter(view_tsne[:,0], view_tsne[:,1], c=pred_l, alpha=0.2, cmap='Set1')
  plt.title(seed+'-tsne-XL',
          fontdict={'family': 'serif',
                    'color' : 'darkblue',
                    #'weight': 'bold',
                    'size': 8})
  plt.savefig(seed+'-tsne-XL.png', dpi=120)


y_true, y_pred = convert_label_(pred_t,labels_t,train_dataset.directories)
confusion = confusion_matrix(y_true, y_pred, labels = train_dataset.directories)
#confusion = confusion_matrix(true_l, pred_l)
print(confusion)
result_model.append("============================= \n")
result_model.append("Confusion_matrix_XL:: \n  "+str(confusion)+ "\n")
plot_confusion_matrix(cm = np.array(confusion),normalize = False, target_names = train_dataset.directories, title = seed+'-XL')


#########90% Unlabeled ##############
print("#XU Unlabeled!")

feature_u= []
labels_u = []
pred_u = []
#img_u = []

#list_metric_unlabeld = []
#new_unlabels_loader = []

running_correct_ = 0
for i, data in enumerate(tqdm(unlabels_loader), 0):
    inputs, labels, indices = data
    inputs  = inputs.to(device)
    labels  = labels.to(device).view(-1)
    #indices = indices.to(device)
    output = model(inputs)
    #neighbours_tr = find_neighbours( num_neighbours, centres, output.data.cpu().numpy() )
    dist_matrix = torch.cdist(output, centres)
    neighbours_tr = torch.argsort(dist_matrix)[:,0:num_neighbours]
    indices_2 = np.arange(0,output.size(0))
    log_prob, prob_real = kernel_classifier( output , centres, centre_labels, neighbours_tr[indices_2, :] )
    pred = log_prob.argmax(dim=1, keepdim=True)
    #writer.add_scalar('Unlabeled_Loss/log_prob',log_prob.mean().data.cpu().numpy(), i)

    #Mcscore_u = MCScore(log_prob).cpu().detach().numpy()
    #entropy_u =  HLoss(prob_real.cpu().detach().numpy())
    #prob_max_u = prob_real.max(1).values.cpu().detach().numpy()

    #for it in range(len(log_prob)):
      #list_metric_unlabeld.append([Mcscore_u[it],entropy_u[it],prob_max_u[it],labels[it].cpu().item(),pred[it].cpu().item()])
      #new_unlabels_loader.append([output[it].data.cpu().numpy(),inputs[it].data.cpu().numpy(),labels[it].cpu().item(),pred[it].cpu().item()])    

    feature_u.append(output.data.cpu().numpy())
    labels_u.append(labels.data.cpu().numpy())
    pred_u.append(pred.data.cpu().numpy())
    #img_u.append(inputs.data.cpu().numpy())
    

    correct = pred.eq(labels.view_as(pred)).sum().item()
    running_correct_ += correct/args.batch_size

#data_U =np.array(list_metric_unlabeld).astype(np.float)
#new_data_U = scaler.transform(data_U[:,0:3]) # normaliza os rotulados 10%
#data_NU = np.array(new_unlabels_loader)

print('####### ACC_pseudo_gaus_labels =',running_correct_/len(unlabels_loader))

result_model.append("============================= \n")
result_model.append("ACC_pseudo_gaus_labels XU::  "+str(running_correct_/len(unlabels_loader))+ "\n")

if(args.tsne_graph == "True"):
  feature_xu,pred_xu,label_xu = unmount_batch_v2(feature_u,pred_u,labels_u)
  view_tsne_xu = TSNE(random_state=123).fit_transform(feature_xu)
  plt.scatter(view_tsne_xu[:,0], view_tsne_xu[:,1], c=pred_xu, alpha=0.2, cmap='Set1')
  plt.title(seed+'-tsne-XU',
          fontdict={'family': 'serif',
                    'color' : 'darkblue',
                    #'weight': 'bold',
                    'size': 8})
  plt.savefig(seed+'-tsne-XU.png', dpi=120)


y_true, y_pred = convert_label_(pred_u,labels_u,train_dataset.directories)
confusion = confusion_matrix(y_true, y_pred, labels = train_dataset.directories)
#confusion = confusion_matrix(true_l, pred_l)
print(confusion)
result_model.append("============================= \n")
result_model.append("Confusion_matrix_XU:: \n  "+str(confusion)+ "\n")
plot_confusion_matrix(cm = np.array(confusion),normalize = False, target_names = train_dataset.directories, title = seed+'-XU')


######### Test  ##############
print("#Test 10k!")

feature_test= []
labels_test = []
pred_test = []
#img_test = []

running_correct_ = 0
for i, data in enumerate(tqdm(test_loader), 0):
    inputs, labels, indices = data
    inputs  = inputs.to(device)
    labels  = labels.to(device).view(-1)
    #indices = indices.to(device)
    output = model(inputs)
    #neighbours_tr = find_neighbours( num_neighbours, centres, output.data.cpu().numpy() )
    dist_matrix = torch.cdist(output, centres)
    neighbours_tr = torch.argsort(dist_matrix)[:,0:num_neighbours]
    indices_2 = np.arange(0,output.size(0))
    log_prob, prob_real = kernel_classifier( output , centres, centre_labels, neighbours_tr[indices_2, :] )
    pred = log_prob.argmax(dim=1, keepdim=True)
    #writer.add_scalar('Unlabeled_Loss/log_prob',log_prob.mean().data.cpu().numpy(), i)

    feature_test.append(output.data.cpu().numpy())
    labels_test.append(labels.data.cpu().numpy())
    pred_test.append(pred.data.cpu().numpy())
    #img_test.append(inputs.data.cpu().numpy())

    correct = pred.eq(labels.view_as(pred)).sum().item()
    running_correct_ += correct/args.batch_size

print('####### ACC_Test_pgl =',running_correct_/len(test_loader))

result_model.append("============================= \n")
result_model.append("ACC_Test::  "+str(running_correct_/len(test_loader))+ "\n")

if(args.tsne_graph == "True"):
  feature_tt,pred_tt, label_tt = unmount_batch_v2(feature_test,pred_test,labels_test)
  view_tsne_u = TSNE(random_state=123).fit_transform(feature_tt)
  plt.scatter(view_tsne_u[:,0], view_tsne_u[:,1], c=label_tt, alpha=0.2, cmap='Set1')
  plt.title(seed+'-tsne_Test',
          fontdict={'family': 'serif',
                    'color' : 'darkblue',
                    #'weight': 'bold',
                    'size': 8})
  plt.savefig(seed+'-tsne_Test.png', dpi=120)


y_true, y_pred = convert_label_(pred_test,labels_test,train_dataset.directories)
confusion = confusion_matrix(y_true, y_pred, labels = train_dataset.directories)
#confusion = confusion_matrix(true_l, pred_l)
print(confusion)
result_model.append("============================= \n")
result_model.append("Confusion_matrix_Test:: \n  "+str(confusion)+ "\n")
plot_confusion_matrix(cm = np.array(confusion),normalize = False, target_names = train_dataset.directories, title = seed+'-Test')


print("############################################################################################################################### ")
"""
print("\n######################################## DML(features,labelling) + RA (RE-labelling) ######################################## ")

acertos_gausian = 0
erros_gaussian = 0
rejeitados = 0
aceitados = 0
correct_metric = 0
erro_metric = 0
num = 0

for i, data in enumerate(tqdm(new_unlabels_loader), 0):
  num += 1
  if((new_data_U[i][0] < (LIMIAR[0] + STD[0])) and (new_data_U[i][1] > (LIMIAR[1] + STD[1])) and (new_data_U[i][2] < (LIMIAR[2] -  STD[2])) ):
    rejeitados +=1 ## rejeitado pelo regra de limiar corretamente. ## nesta parte poder conter mais rotulos errado de que acertado pelo gausiano
    featureU,imgU,true_label,labelU = data
    #feature_,imgs_,label_,correct_,erro_ = pairwise_distances_(featureU,imgU,labelU,feature_l,img_l,label_l,true_label)
    correct_,erro_ = pairwise_distances_(featureU,imgU,labelU,feature_l,img_l,label_l,true_label, args.topk)

    #save_image(torch.from_numpy(np.array(imgs_)), path+'/'+label_names[label_[0]]+'/img_xr_'+str(label_[0])+'_'+str(i)+'_R_'+str(num)+'.png',normalize=True)

    correct_metric = correct_metric + correct_
    erro_metric = erro_metric + erro_

    if(true_label == labelU): # se o rotulo real é igual ao rotulo dado pela distancia
      acertos_gausian += 1 ## rejeitado pela regra mas esta errado pois o pseudo lebel da gaus acertou 

  else:
    featureU_acept,imgsU_acept,true_label,labelU_acept = data
    #save_image(torch.from_numpy(np.array(imgsU_acept)), path+'/'+label_names[labelU_acept]+'/img_xr_'+str(labelU_acept)+'_'+str(i)+'_A_'+str(num)+'.png',normalize=True)

    aceitados += 1 ##aceito corretamente
    if(true_label != labelU_acept):
      erros_gaussian += 1 ## foi aceito mas esta errado



print("Aceitados  ; ",aceitados)
print("Aceitados que o ML ACERTOU (TP) : ",(aceitados-erros_gaussian))
print("Aceitados que o ML ERROU (FP) : ",erros_gaussian)

print("\nRejeitados ; ",rejeitados)
print("Rejeitados que RA ACERTOU (TN) :",correct_metric)
print("Rejeitados que RA ERROU (FN) : ",erro_metric)

print("ACC: ",( (aceitados-erros_gaussian) + correct_metric ) / new_data_U.shape[0])

result_model.append("============================= \n")
result_model.append("DML(features,labelling) + RA (RE-labelling) \n")
result_model.append("ACC_Test::  "+str(((aceitados-erros_gaussian) + correct_metric ) / new_data_U.shape[0])+ "\n")

print("\n######################################## DML(features,labelling) ######################################## ")

acertos_gausian = 0
erros_gaussian = 0
rejeitados = 0
aceitados = 0
correct_metric = 0
erro_metric = 0
num = 0

for i, data in enumerate(tqdm(new_unlabels_loader), 0):
    num += 1
    featureU_acept,imgsU_acept,true_label,labelU_acept = data
    aceitados += 1 ##aceito corretamente
    if(true_label != labelU_acept):
      erros_gaussian += 1 ## foi aceito mas esta errado

print("Aceitados  ; ",aceitados)
print("Aceitados que o ML ACERTOU (TP) : ",(aceitados-erros_gaussian))
print("Aceitados que o ML ERROU (FP) : ",erros_gaussian)
print("ACC: ", (aceitados-erros_gaussian) / new_data_U.shape[0] )

result_model.append("============================= \n")
result_model.append("DML(features,labelling) \n")
result_model.append("ACC_Test::  "+str((aceitados-erros_gaussian) / new_data_U.shape[0])+ "\n")


print("\n######################################## DML(features) + RA (labelling) ######################################## ")

acertos_gausian = 0
erros_gaussian = 0
rejeitados = 0
aceitados = 0
correct_metric = 0
erro_metric = 0
num = 0

for i, data in enumerate(tqdm(new_unlabels_loader), 0):
    num += 1
    rejeitados +=1 ## rejeitado pelo regra de limiar corretamente. ## nesta parte poder conter mais rotulos errado de que acertado pelo gausiano
    featureU,imgU,true_label,labelU = data
    correct_,erro_ = pairwise_distances_(featureU,imgU,labelU,feature_l,img_l,label_l,true_label, args.topk)
    correct_metric = correct_metric + correct_
    erro_metric = erro_metric + erro_

print("\nRejeitados ; ",rejeitados)
print("Rejeitados que RA ACERTOU (TN) :",correct_metric)
print("Rejeitados que RA ERROU (FN) : ",erro_metric)

print("ACC: ", correct_metric / new_data_U.shape[0])

result_model.append("============================= \n")
result_model.append("DML(features) + RA (labelling) \n")
result_model.append("ACC_Test::  "+str(correct_metric / new_data_U.shape[0])+ "\n")

"""

arquivo = open(seed+".txt", "a")
arquivo.writelines(result_model)
arquivo.close()
print("finished")
