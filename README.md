# Weekly-Learning_DML-Mixup_GAN

python3 train_fe_DML_OUR_mixup.py --max_epochs=200 --alpha 1 --scale_mixup 0.1 --beta 1 --topk 3 --data_dir cifar10_train10%_labels  --unlabels cifar10_train90%_unlabels --test cifar10_test100%_labels --save_dir /results/neighbour=200 --num_classes 10 tsne_print=False --im_ext png --gpu_id 0 --input_size 32
