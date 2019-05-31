#!/bin/bash

cd ..
#source activate pytorch-env

NAME=10_mnist_baseline_g16_frontal_bid_lamb10_inorm_32d_kyle_faithful
python task_launcher.py \
--dataset=mnist \
--arch=architectures/arch_acai_kyle.py \
--save_every=50 \
--epochs=500 \
--resume=auto \
--n_channels=1 \
--ngf=0 \
--ndf=0 \
--name=${NAME} \
--data_dir=/tmp/beckhamc/img_align_celeba \
--batch_size=16 \
--beta=0.0 \
--lamb=10.0 \
--cls=0.0 \
--save_path=results_mixup \
--mixer=mixup \
--disable_mix \
--seed=1 \
--classify_encoding='32,10' \
--weight_decay=1e-5 \
--beta1=0.5 \
--beta2=0.99 \
--lr=1e-4
