#!/bin/bash

# Run this from the ./exps folder.

NAME=mnist_baseline
cd ..
python task_launcher.py                  \
--dataset=mnist                          \
--arch=architectures/arch_acai_kyle.py   `# This architecture is from the ACAI repo`\
--save_every=100                         `# Save model every 100 epochs`\
--save_images_every=100                  \
--save_path=results                      \
--epochs=1000                            \
--resume=auto                            \
--n_channels=1                           `# MNIST is black and white`\
--ngf=0                                  `# arch_acai_kyle.py ignores ngf, so this can be anything`\
--ndf=0                                  `# arch_acai_kyle.py ignores ndf, so this can be anything`\
--name=${NAME}                           \
--batch_size=64                          \
--beta=0.0                               `# no consistency loss`\
--lamb=10.0                              `# reconstruction weight`\
--cls=0.0                                `# we're not doing supervised mixes`\
--disable_mix                            `# disable mixup -- this reduces the class into an adversarial AE`\
--mixer=mixup                            `# mixing func -- ignored due to disable_mix above`\
--seed=${SEED}                           `# if seed is e.g. 5, then the experiment is saved in results/s5`\
--classify_encoding='32,10'              `# bottleneck is 32 units, train a classifier over it which predicts 10 classes`\
--weight_decay=1e-5                      `# L2 norm on the weights`\
--beta1=0.5                              `# beta1 for ADAM optimiser`\
--beta2=0.99                             `# beta2 for ADAM optimiser`\
--lr=1e-4                                `# learning rate for ADAM`
