#!/bin/bash

cd ..
#source activate pytorch-env

NAME=svhn32-20000_mixup3
python task_launcher.py \
--model=threegan \
--dataset=svhn \
--arch=architectures/arch_acai_kyle.py \
--save_every=100 \
--save_images_every=100 \
--epochs=4000 \
--resume=auto \
--n_channels=3 \
--ngf=0 \
--ndf=0 \
--name=${NAME} \
--batch_size=64 \
--beta=0.0 \
--lamb=10.0 \
--cls=0.0 \
--save_path=results_mixup \
--mixer=mixup \
--seed=1 \
--cls_probe=architectures/cls_probes/linear_legacy.py \
--cls_probe_args="{'n_in':32, 'n_classes':10}" \
--weight_decay=1e-5 \
--beta1=0.5 \
--beta2=0.99 \
--lr=1e-4 \
--subset_train=20000
