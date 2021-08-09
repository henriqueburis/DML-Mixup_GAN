# Weekly-Learning_DML-Mixup_GAN

##Train##

python3 train_fe_DML_OUR_mixup.py --max_epochs=200 --scale_mixup 0.1 --alpha 1 --beta 1 --topk 3 --data_dir cifar10_train10%_labels  --unlabels cifar10_train90%_unlabels --test cifar100_test10%_labels --save_dir results/neighbour=200 --num_classes 10 --tsne_graph True --im_ext png --gpu_id 0 --input_size 32

##Test##

python3 test_dml_mixup.py --max_epochs=200 --scale_mixup 0.1 --alpha 1 --beta 1 --topk 3 --data_dir cifar10_train10%_labels --unlabels cifar10_train90%_unlabels --test cifar10_test100%_labels --load_dir results/neighbour=200 --num_classes 10 --im_ext png --tsne_graph False --gpu_id 0 --input_size 32
