#!/bin/bash

cd ..
#source activate pytorch-env

python task_launcher.py run \
--dataset=svhn \
--arch=architectures/arch_acai_kyle.py \
--save_every=100 \
--save_images_every=100 \
--epochs=6000 \
--resume=auto \
--n_channels=3 \
--ngf=0 \
--ndf=0 \
--batch_size=64 \
--beta=0.0 \
--lamb=10.0 \
--cls=0.0 \
--mixer=mixup \
--seed=1 \
--cls_probe=architectures/cls_probes/linear_legacy.py \
--cls_probe_args="{'n_in':32, 'n_classes':10}" \
--weight_decay=1e-5 \
--beta1=0.5 \
--beta2=0.99 \
--lr=1e-4 \
--subset_train=10000
