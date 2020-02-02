#!/bin/bash

cd ..
#source activate pytorch-env

python task_launcher.py run \
--model=kgan \
--k=6 \
--dataset=cifar10 \
--arch=architectures/arch_acai_kyle_256.py \
--save_every=10 \
--save_images_every=10 \
--epochs=3000 \
--resume=auto \
--n_channels=3 \
--ngf=0 \
--ndf=0 \
--batch_size=64 \
--beta=0.0 \
--lamb=50.0 \
--cls=0.0 \
--mixer=mixup \
--seed=1 \
--cls_probe=architectures/cls_probes/linear_legacy.py \
--cls_probe_args="{'n_in': 256, 'n_classes':10}" \
--weight_decay=1e-5 \
--beta1=0.5 \
--beta2=0.99 \
--lr=1e-4
