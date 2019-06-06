# Adversarial mixup resynthesizers

<img src="https://github.com/christopher-beckham/amr/raw/dev/figures/mixup_anim.gif" width=225 /> <img src="https://github.com/christopher-beckham/amr/raw/dev/figures/mixup3_anim.gif" width=225 /> <img src="https://github.com/christopher-beckham/amr/raw/dev/figures/fm_anim.gif" width=225 />

In this paper, we explore new approaches to combining information encoded within the learned representations of autoencoders. We explore models that are capable of combining the attributes of multiple inputs such that a resynthesised output is trained to fool an adversarial discriminator for real versus synthesised data. Furthermore, we explore the use of such an architecture in the context of semi-supervised learning, where we learn a mixing function whose objective is to produce interpolations of hidden states, or masked combinations of latent representations that are consistent with a conditioned class label. We show quantitative and qualitative evidence that such a formulation is an interesting avenue of research.

<img src="https://github.com/christopher-beckham/amr/raw/dev/figures/model.png" width=768px />

## Setting up the project

### Cloning the repository:
`$ git clone <insert url here>`

### Environment setup

1. Install Anaconda, if not already done, by following these instructions:
https://docs.anaconda.com/anaconda/install/linux/  

2. Create a conda environment using the `environment.yaml` file, to install the dependencies:  
`$ conda env create -f environment.yaml`

3. Activate the new conda environment:
`$ conda activate amr`

### Getting the data

For this iteration of the code, there is no need to download external datasets since we will be using the ones provided with `torchvision` (namely, MNIST and SVHN).

## Running experiments

The experiment scripts can be found in the `exps` folder. Simply `cd` into this folder and run `bash <folder_name>/<script_name>.sh`. Experiments for Table 1
in the paper correspond to the folders `mnist_downstream`, `kmnist_downstream`, and `svhn32_downstream`. For Table 3, consult `svhn256_downstream`.

### Training the models

In order to launch experiments, we use the `task_launcher.py` script. This is a bit hefty at this point and contains a lot of argument options,
so it's recommended you get familiar with them by running `python task_launcher.py --help`. To help you get started, we present an example
script corresponding to running a simple baseline where we train an adversarial reconstruction autoencoder (ARAE) on MNIST. Copy and paste this
example into a bash script inside the `exps` folder.

```
NAME=mnist_baseline
cd ..
python task_launcher.py                  \
--dataset=mnist                          \
--arch=architectures/arch_acai_kyle.py   `# This architecture is derived from https://gist.github.com/kylemcdonald/e8ca989584b3b0e6526c0a737ed412f0`\
--save_every=100                         `# Save model every 100 epochs`\
--save_images_every=100                  \
--save_path=results                      \
--epochs=1000                            \
--resume=auto                            `# When set to 'auto', the script will automatically load the latest checkpt`\
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
--seed=1                                 `# if seed is e.g. s1, then the experiment is saved in results/s1`\
--weight_decay=1e-5                      `# L2 norm on the weights`\
--beta1=0.5                              `# beta1 for ADAM optimiser`\
--beta2=0.99                             `# beta2 for ADAM optimiser`\
--lr=1e-4                                `# learning rate for ADAM` \
```

If we wanted to evaluate the performance of features extracted on the bottleneck on MNIST classification, we simply add
these lines to the script:

```
--cls_probe=architectures/cls_probes/linear.py `# logistic regression classifier` \
--cls_probe_args="{'n_in': 32, 'n_classes':10}" `# the bottleneck of arch_acai_kyle.py is of dimension 4x4x2, so it is == 32 when flattened`\
```

What if we wanted to turn this model into AMR? Easy! Simply remove `--disable_mix` and now we have AMR with the mixup function. If we change
`--mixer=mixup` to `--mixer=fm2`, then we have AMR with Bernoulli mixup. For mixing in triplets, simply add `--model=threegan`.

### Evaluating samples

This also easy! Simply add `--mode=interp_train` (or `--mode=interp_valid`) to the script. This changes the mode in the task launcher script
from training (which is the default) to interpolation mode. In this mode, interpolations between samples will be produced and output in the
results folder. The number of samples used for interpolation is dependent on `--val_batch_size`.

## Notes

- The main architecture we use here is one derived from a PyTorch reimplementation of ACAI, courtesy of Kyle McDonald, whose implementation can be found here: https://gist.github.com/kylemcdonald/e8ca989584b3b0e6526c0a737ed412f0
  - The main changes we make is that we add spectral norm to the discriminator to stabilise GAN training. We also added instance norm to the generator to stabilise training.
  - Generator code: https://github.com/christopher-beckham/amr/blob/dev/architectures/arch_kyle.py#L21-L96
  - Discriminator code: https://github.com/christopher-beckham/amr/blob/dev/architectures/arch_kyle.py#L98-L108
- ACAI's regularisation term has not been implemented, so it's not a completely faithful reproduction of their model. We will address this issue.
- Having batch norm in the generator appears to generate funky artifacts (or at least it was the case on our qualitative experiments on CelebA/Zappos in the paper). Instead we opted for instance norm.

## Troubleshooting

If you are experiencing any issues, please file a ticket in the Issues section.
