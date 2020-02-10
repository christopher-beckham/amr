#!/bin/bash

cd ..
#source activate pytorch-env

python task_launcher.py run \
--model=threegan \
--dataset=dsprite \
--arch=architectures/dsprite_gd.py \
--save_every=2 \
--save_images_every=1 \
--epochs=500 \
--ndf=32 \
--resume=auto \
--n_channels=1 \
--batch_size=64 \
--beta=0.0 \
--lamb=1.0 \
--cls=0.0 \
--mixer=fm \
--seed=4 \
--beta1=0.5 \
--beta2=0.99 \
--lr=1e-4 \
--recon_loss=bce_sum
#--mode=dsprite_disentangle_fv
#--mode_override='num_examples=10000'
