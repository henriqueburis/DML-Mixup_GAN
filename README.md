# Weekly-Learning_DML-Mixup_GAN


# Code organization

- `train_fe_DML_OUR_mixup.py`: main command line interface and training loops.
- `test_dml_mixup.py` : test.


# Command examples

Show all the program arguments
```
python3 train_fe_DML_OUR_mixup.py --max_epochs=200 --scale_mixup 0.1 --alpha 1 --beta 1 --topk 3 --data_dir cifar10_train10%_labels  --unlabels cifar10_train90%_unlabels --test cifar10_test100%_labels --save_dir results/neighbour=200 --num_classes 10 --tsne_graph False --im_ext png --gpu_id 0 --input_size 32

python3 test_dml_mixup.py --name "CIFAR10" --max_epochs=200 --scale_mixup 0 --alpha 1 --beta 1 --topk 3 --data_dir CIFAR10/XL  --unlabels CIFAR10/XU --test CIFAR10/Test --load_dir results/neighbour=200 --num_classes 10 --tsne_graph True --im_ext png --gpu_id 0 --input_size 32
```


## Wide Resnet50 on the original CIFAR-10
```
python train.py
```
and here are some sample outputs from my local run:
```

```

## Wide Resnet50 on CIFAR-10 with Mixup loss
```
python trainxxxxxx with mixup loss
```
and here are some sample outputs from my local run:



# Mixup loss

In Section 5 of the paper, we talked about SGD implicit regularizing by finding the minimum norm solution in an over-parameterized linear problem with the simple square loss. A frequently asked question is how the experiments on MNIST is conducted, since it has 60,000 training examples, but only 28x28 = 784 features. As mentioned in the paper, the experiments are actually carried out by applying the "kernel trick" to Equation (3) in the paper (for both MNIST and CIFAR-10, with or without pre-processing). We are attaching a sample code for solving the MNIST raw pixel problem for reference:

![N|Solid](https://github.com/henriqueburis/Weekly-Learning_DML-Mixup_GAN/blob/main/figure/CIFAR10-EP200-SM5.0-A1.0-B1.0-XL-confusion_matrix.png?raw=true)
![N|Solid](https://github.com/henriqueburis/Weekly-Learning_DML-Mixup_GAN/blob/main/figure/CIFAR10-EP200-SM5.0-A1.0-B1.0-XU-confusion_matrix.png?raw=true)
![N|Solid](https://github.com/henriqueburis/Weekly-Learning_DML-Mixup_GAN/blob/main/figure/CIFAR10-EP200-SM5.0-A1.0-B1.0-tsne-XL.png?raw=true)
![N|Solid](https://github.com/henriqueburis/Weekly-Learning_DML-Mixup_GAN/blob/main/figure/CIFAR10-EP200-SM5.0-A1.0-B1.0-tsne-XU.png?raw=true)

![N|Solid](https://github.com/henriqueburis/Weekly-Learning_DML-Mixup_GAN/blob/main/figure/CIFAR100%20-%20XL10%25%20ACC%20-%20accuracy.png?raw=true)
![N|Solid](https://github.com/henriqueburis/Weekly-Learning_DML-Mixup_GAN/blob/main/figure/CIFAR100%20-%20XL10%25%20ACC%20-%20test.png?raw=true)

```python


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
print("loss_mixup",loss_mixup)
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


```
