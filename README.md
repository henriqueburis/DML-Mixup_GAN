# Weekly-Learning_DML-Mixup_GAN


# Code organization

- `train_fe_DML_OUR_mixup.py`: main command line interface and training loops.
- `test_dml_mixup.py` : test.


# Command examples

Show all the program arguments
```
python3 train_fe_DML_OUR_mixup.py --max_epochs=200 --scale_mixup 0.1 --alpha 1 --beta 1 --topk 3 --data_dir cifar10_train10%_labels  --unlabels cifar10_train90%_unlabels --test cifar10_test100%_labels --save_dir results/neighbour=200 --num_classes 10 --tsne_graph False --im_ext png --gpu_id 0 --input_size 32

python3 test_dml_mixup.py --max_epochs=200 --scale_mixup 0.1 --alpha 1 --beta 1 --topk 3 --data_dir cifar10_train10%_labels --unlabels cifar10_train90%_unlabels --test cifar10_test100%_labels --load_dir results/neighbour=200 --num_classes 10 --im_ext png --tsne_graph True --gpu_id 0 --input_size 32
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

```python
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

import numpy as np
import scipy.spatial
import scipy.linalg

def onehot_encode(y, n_cls=10):
    y = np.array(y, dtype=int)
    return np.array(np.eye(n_cls)[y], dtype=float)

data = np.array(mnist.data, dtype=float) / 255
labels = onehot_encode(mnist.target)

n_tr = 60000
n_tot = 70000

x_tr = data[:n_tr]; y_tr = labels[:n_tr]
x_tt = data[n_tr:n_tot]; y_tt = labels[n_tr:n_tot]

bw=2.0e-2
pdist_tr = scipy.spatial.distance.pdist(x_tr, 'sqeuclidean')
pdist_tr = scipy.spatial.distance.squareform(pdist_tr)
cdist_tt = scipy.spatial.distance.cdist(x_tt, x_tr, 'sqeuclidean')

coeff = scipy.linalg.solve(np.exp(-bw*pdist_tr), y_tr)
preds = np.argmax(np.dot(np.exp(-bw*cdist_tt), coeff), axis=1)

acc = float(np.sum(np.equal(preds, mnist.target[n_tr:n_tot]))) / (n_tot-n_tr)
print('err = %.2f%%' % (100*(1-acc)))

# err = 1.22%
```
