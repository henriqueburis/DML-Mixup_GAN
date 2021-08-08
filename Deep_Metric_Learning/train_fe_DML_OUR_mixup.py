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

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.autograd import Variable

import numpy as np
import os
import subprocess
import argparse
import scipy

from copy import deepcopy
from sklearn.manifold import TSNE

import auto_augment as ag
from utils import *


######################## imports Utils ############################################
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,
KNeighborsClassifier)
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
#from scipy.spatial.distance.cdist import distance
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


parser = argparse.ArgumentParser(description="Train Gaussian kernel classifier using Resnet18 or 50.")
parser.add_argument("--data_dir", required=True, type=str, help="Path to data parent directory.")
parser.add_argument("--unlabels", required=True, type=str, help="Path to data parent directory.")
parser.add_argument("--test", required=True, type=str, help="Path to data parent directory.")
parser.add_argument("--save_dir", required=True, type=str, help="Models are saved to this directory.")
parser.add_argument("--num_classes", required=True, type=int, help="Number of training classes to use.")

parser.add_argument("--im_ext", default="jpg", type=str, help="Dataset image file extensions (e.g. jpg, png).")
parser.add_argument("--gpu_id", default=None, type=int, help="GPU ID. CPU is used if not supplied.")
parser.add_argument("--sigma", default=10, type=int, help="Gaussian sigma.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--learning_rate", default=1e-5, type=int, help="learning_rate")
parser.add_argument("--update_interval", default=5, type=int, help="Stored centres/neighbours are updated every update_interval epochs.")
parser.add_argument("--max_epochs", default=50, type=int, help="Maximum training length (epochs).")
parser.add_argument("--topk", default=20, type=int, help="top k.")
parser.add_argument("--input_size", default=256, type=int, help="input size img.")
####MIXUP
parser.add_argument('--alpha', default=1, type=float,help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--scale_mixup', default=0.0001, type=float,help='scaling the mixup loss')
parser.add_argument('--beta', default=1, type=float,help='scaling the gauss loss')
args = parser.parse_args()


args = parser.parse_args()
#print("/run/-"+args.data_dir+"-"+str(args.max_epochs)+"ep-scale_mixup "+str(args.scale_mixup)+"-alpha"+str(args.alpha))
writer = SummaryWriter(comment="-"+args.data_dir+"-"+str(args.max_epochs)+"ep-scale_mixup "+str(args.scale_mixup)+"-alpha"+str(args.alpha)+"-beta"+str(args.beta))

seed = str(args.max_epochs)+str(args.scale_mixup)+str(args.alpha)+str(args.beta)
print('seed==>',seed) 

"""
Configuration
"""

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

kernel_weights_lr = args.learning_rate*1
num_neighbours    = 200
eval_interval     = args.update_interval

#Set GPU ID or 'cpu'
if args.gpu_id is None:
	device = torch.device('cpu')
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
	device = torch.device('cuda:0')


import torch.nn.functional as F

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        #print(x)
        #b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = x*np.log(x)
        b = np.sum(b,axis=1)
        #b = -1.0 * b.sum()
        b = -1.0 * b
        return b


HLoss = HLoss()

"""
Set up DataLoaders
"""

#Transformations/pre-processing operations
train_transforms = transforms.Compose([
        transforms.Resize(input_size),
#        transforms.RandomCrop(input_size),
#        ag.AutoAugment(),
#        ag.Cutout(),
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
#best_state = model.to(device)
#model = model.train()
model = model.eval()

def update_centres():

	#Disable dropout, use global stats for batchnorm
	model.eval()

	#Disable learning
	with torch.no_grad():

		#Update stored centres
		for i, data in enumerate(update_loader, 0):

			# Get the inputs; data is a list of [inputs, labels]. Send to GPU
			inputs, labels, indices = data
			inputs = inputs.to(device)

			#Extract features for batch
			extracted_features = model(inputs)
			#print(extracted_features.shape[0])

			#Save to centres tensor
			idx = i*args.batch_size
			centres[idx:idx + extracted_features.shape[0], :] = extracted_features

	#model.train()
	model.eval()

	return centres


def save_model():
	torch.save(model.state_dict(), args.save_dir + "/"+seed+"model.pt")
	torch.save(kernel_classifier.state_dict(), args.save_dir + "/"+seed+"classifier.pt")
	torch.save(centres, args.save_dir + "/"+seed+"centres.pt")

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


"""
Set up loss and optimiser
"""

criterion = nn.NLLLoss()

optimiser = optim.Adam([
                {'params': model.parameters()},
                {'params': kernel_classifier.parameters(), 'lr': kernel_weights_lr}
            ], lr=args.learning_rate)

#exp_lr_scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=step_size, gamma=step_gamma)

##################################################### MIXUP #######################################################

criterion_mixup = nn.CrossEntropyLoss()

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

############################################################################ FIM ##############################################################

"""
 Test
"""

def test():
    print("Test!")
    #model = model.eval()
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

        correct = pred.eq(labels.view_as(pred)).sum().item()
        running_correct_ += correct/args.batch_size

    acc = running_correct_/len(unlabels_loader)

    print('####### ACC_Test_train =',acc)
    return acc


"""
Training
"""
print("Begin training...")
acc_geral = -1
for epoch in range(args.max_epochs):  # loop over the dataset multiple times

	#Update stored kernel centres
	if (epoch % args.update_interval) == 0:

		print("Updating kernel centres...")
		centres = update_centres()
		print("Finding training set neighbours...")
		centres = centres.cpu()
		neighbours_tr = find_neighbours( num_neighbours, centres )
		centres = centres.to(device)
		print("Finished update!")

		if epoch > 0:
                    acc_ataual = test()
                    writer.add_scalar('ACC/test', acc_ataual, epoch)
                    if(acc_geral <= acc_ataual):
                       #print('-------------------------------------->',acc_ataual)
                       acc_geral = acc_ataual
                       #best_state = model.state_dict()
                       save_model()
                    #test()

	#Training
	running_loss = 0.0
	running_correct = 0
	for i, data in enumerate(train_loader, 0):
		# Get the inputs; data is a list of [inputs, labels]. Send to GPU
                inputs, labels, indices = data
                inputs  = inputs.to(device)
                labels  = labels.to(device).view(-1)
                indices = indices.to(device)

                ######################################## MIXUP
                inputs_mixup, targets_a, targets_b, lam = mixup_data(inputs, labels,args.alpha, True)
                inputs_mixup, targets_a, targets_b = map(Variable, (inputs_mixup,targets_a, targets_b))
                outputs_mixup,outputs_2 = kernel_classifier( model(inputs_mixup), centres, centre_labels, neighbours_tr[indices, :] )
                loss_mixup = mixup_criterion(criterion_mixup, outputs_mixup, targets_a, targets_b, lam) #MIXUP loss
                #print("loss_mixup",loss_mixup)
                ##############################################

		# Zero the parameter gradients
                optimiser.zero_grad()

                log_prob, prob_real = kernel_classifier( model(inputs), centres, centre_labels, neighbours_tr[indices, :])

                loss_gauss = criterion(log_prob, labels) # gaussian loss
		#scale_mixup = 0.01
                loss = (args.beta * loss_gauss) + (args.scale_mixup * loss_mixup)

                #loss =  criterion(log_prob,labels)
                
                loss.backward()
                optimiser.step()
                
                running_loss += loss.item()
                writer.add_scalar('Loss/loss_gauss', loss_gauss, (epoch*len(train_loader.dataset)/32)+i)
                writer.add_scalar('Loss/loss_mixup', loss_mixup, (epoch*len(train_loader.dataset)/32)+i)
                writer.add_scalar('Loss/loss', loss, (epoch*len(train_loader.dataset)/32)+i)
                
                #Get the index of the max log-probability
                pred = log_prob.argmax(dim=1, keepdim=True)
                correct = pred.eq(labels.view_as(pred)).sum().item()
                running_correct += correct

	#Print statistics at end of epoch
	if True:
		print('[{0}, {1:5d}] loss: {2:.3f}, accuracy: {3}/{4} ({5:.4f}%)'.format(
			epoch + 1, i + 1, running_loss / len(train_loader.dataset),
                        running_correct, len(train_loader.dataset), 100. * running_correct / len(train_loader.dataset)))
		writer.add_scalar('ACC/accuracy', 100. * running_correct / len(train_loader.dataset), (epoch*len(train_loader.dataset)/32)+i)
		running_loss = 0.0
		running_correct = 0

        #exp_lr_scheduler.step()

#Update centres final time when done
print("Updating kernel centres (final time)...")
centres = update_centres()
#print("The best acc=====================",acc_geral)
writer.add_scalar('ACC/acc_geral', acc_geral, 1)
#save_model()


"""
#path = './train'
if(args.num_classes == 10):
  #path = './'+args.save_dir+'/cifar'+args.num_classes+''
  path = './cifar'+str(args.num_classes)+''
  #CreateDir(path)
  label_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  #for i in range(len(label_names)):
  #  CreateDir(path+'/'+label_names[i]+'/')
else:
  path = './cifar'+str(args.num_classes)+''
 # CreateDir(path)
  label_names=['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
  #for i in range(len(label_names)):
  #  CreateDir(path+'/'+label_names[i]+'/')
"""

def MCScore(log_prob):
  top2 = torch.topk(log_prob, k=2, dim=1).values[:,1]
  top1 = torch.topk(log_prob, k=2, dim=1).values[:,0]
  score_c = (top2 - top1) / torch.sum(log_prob,dim=1)
  return score_c

def TopK(final_dist,label_l,true_labels, k):
  final_dist = np.squeeze(np.array(final_dist))
  k = 20
  freq = np.zeros(100)
  topk = np.sort(final_dist)[0:k] # select top k

  for i in range(topk.shape[0]):
     pos = np.where(topk[i]==final_dist)[0]
     freq[label_l[pos]] += 1

  final_index = np.argmax(freq)
  return  final_index

def pairwise_distances_(feature_u, img_u, label_u, feature_l, img_l, label_l, true_labels):
  labels = []
  correct = 0
  erro = 0

  dist_matrix_1 = pairwise_distances(np.array([feature_u]),feature_l , metric = 'euclidean')  #dist_matrix_1.shape(1,5000)
  dist_matrix_scaler_1 = (dist_matrix_1 - dist_matrix_1.min()) / (dist_matrix_1.max() - dist_matrix_1.min())

  dist_matrix_4 = pairwise_distances(np.array([feature_u]),feature_l , metric = 'chebyshev')
  dist_matrix_scaler_4 = (dist_matrix_4 - dist_matrix_4.min()) / (dist_matrix_4.max() - dist_matrix_4.min())

  dist_matrix_5 = pairwise_distances(np.array([feature_u]),feature_l , metric = 'cityblock')
  dist_matrix_scaler_5 = (dist_matrix_5 - dist_matrix_5.min()) / (dist_matrix_5.max() - dist_matrix_5.min())

  final_dist = (1+dist_matrix_scaler_1) * (1+dist_matrix_scaler_5) * (1+dist_matrix_scaler_4)
  
  k = args.topk
  final_index = TopK(final_dist,label_l, true_labels,k)

  if(true_labels == final_index): #label_l[final_index]):
    correct = correct +1;
  else:
    erro = erro + 1

  return correct,erro

print("########################################################################################")
print("########################################################################################")

############################ Load best state model ######################################
#model = best_state.eval()
model.load_state_dict(torch.load('./results/neighbour=200/'+seed+'model.pt',map_location=device))
model = model.eval()
########################################################################################

#######10% labeled ##############
print("#XL labeled")

feature_t= []
labels_t = []
img_t = []
running_correct = 0
list_metric_labeld = []
for i, data in enumerate(tqdm(train_loader), 0):
    inputs, labels, indices = data
    inputs  = inputs.to(device)
    labels  = labels.to(device).view(-1)
    indices = indices.to(device)
    output = model(inputs)
    log_prob, prob_real = kernel_classifier( output , centres, centre_labels, neighbours_tr[indices, :] )
    pred = log_prob.argmax(dim=1, keepdim=True)

    Mcscore_l = MCScore(log_prob).cpu().detach().numpy()
    entropy_l =  HLoss(prob_real.cpu().detach().numpy())
    prob_max_l = prob_real.max(1).values.cpu().detach().numpy()

    for it in range(len(log_prob)):
      list_metric_labeld.append([Mcscore_l[it],entropy_l[it],prob_max_l[it],labels[it].cpu().item(),pred[it].cpu().item()])

    feature_t.append(output.data.cpu().numpy())
    labels_t.append(labels.data.cpu().numpy())
    img_t.append(inputs.data.cpu().numpy())
    correct = pred.eq(labels.view_as(pred)).sum().item()
    running_correct += correct/args.batch_size

"""
data_L = np.array(list_metric_labeld).astype(np.float)
scaler.fit(data_L[:,0:3])
new_data_L = scaler.transform(data_L[:,0:3]) # normaliza os rotulados 10%

LIMIAR = np.mean(new_data_L,axis=0) # limiares definidos pela média dos 10% rotulados em cima da média das três métricas
STD = np.std(new_data_L, axis=0)
print("LIMIAR XL",LIMIAR)
print("STD XL",STD)
"""

print('####### AAC_Label = ',running_correct/len(train_loader))
feature_l,img_l, label_l = unmount_batch(feature_t,img_t,labels_t)

view_tsne = TSNE(random_state=123).fit_transform(feature_l)
plt.scatter(view_tsne[:,0], view_tsne[:,1], c=label_l, alpha=0.2, cmap='Set1')
plt.savefig(args.data_dir+'-'+str(args.max_epochs)+'-ep-scale_mixup-'+str(args.scale_mixup)+'-alpha-'+str(args.alpha)+'-tsne-XL.png', dpi=120)


#########90% Unlabeled ##############
print("#XU Unlabeled!")

feature_u= []
labels_u = []
img_u = []

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

    feature_u.append(output.data.cpu().numpy())
    labels_u.append(labels.data.cpu().numpy())
    img_u.append(inputs.data.cpu().numpy())

    correct = pred.eq(labels.view_as(pred)).sum().item()
    running_correct_ += correct/args.batch_size

print('####### ACC_pseudo_gaus_labels =',running_correct_/len(unlabels_loader))

feature_xu,img_xu, label_xu = unmount_batch(feature_u,img_u,labels_u)

view_tsne_xu = TSNE(random_state=123).fit_transform(feature_xu)
plt.scatter(view_tsne_xu[:,0], view_tsne_xu[:,1], c=label_xu, alpha=0.2, cmap='Set1')
plt.savefig(args.data_dir+'-'+str(args.max_epochs)+'-ep-scale_mixup-'+str(args.scale_mixup)+'-alpha-'+str(args.alpha)+'-tsne-XU.png', dpi=120)


######### Test  ##############
print("#Test 10k!")

feature_test= []
labels_test = []
img_test = []

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
    img_test.append(inputs.data.cpu().numpy())

    correct = pred.eq(labels.view_as(pred)).sum().item()
    running_correct_ += correct/args.batch_size

print('####### ACC_Test_pgl =',running_correct_/len(unlabels_loader))

feature_tt,img_tt, label_tt = unmount_batch(feature_test,img_test,labels_test)

view_tsne_u = TSNE(random_state=123).fit_transform(feature_tt)
plt.scatter(view_tsne_u[:,0], view_tsne_u[:,1], c=label_tt, alpha=0.2, cmap='Set1')
plt.savefig(args.data_dir+'-'+str(args.max_epochs)+'-ep-scale_mixup-'+str(args.scale_mixup)+'-alpha-'+str(args.alpha)+'-beta-'+str(args.beta)+'-tsne_Test.png', dpi=120)



print("finished")
