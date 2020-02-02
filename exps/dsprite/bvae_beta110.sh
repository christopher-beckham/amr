#!/bin/bash

cd ..
#source activate pytorch-env

python task_launcher.py run \
--model=vae \
--dataset=dsprite \
--arch=architectures/arch_dsprite_burgess-g_kyle_d_vae.py \
--save_every=10 \
--save_images_every=1 \
--epochs=500 \
--resume=auto \
--n_channels=1 \
--ndf=16 \
--name=${NAME} \
--batch_size=64 \
--beta=110.0 \
--lamb=1.0 \
--cls=0.0 \
--mixer=mixup \
--seed=4 \
--beta1=0.9 \
--beta2=0.99 \
--lr=1e-4 \
--recon_loss=bce
