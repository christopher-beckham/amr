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

(Note: this was my dev environment exported directly to yaml, and may contain a lot of unnecessary dependencies. If you want a more clean environment, it shouldn't be too hard to start from scratch -- all of the dependencies can be easily downloaded with either `pip` or `conda`.)

### Getting the data

For most of the experiments, there is no need to download external datasets since they are already provided `torchvision` (namely, MNIST and SVHN). The exception to this is the DSprites dataset (used for the disentanglement experiments). In order to download this, simply do:

```
cd iterators
wget https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
```

## Running experiments

The experiment scripts can be found in the `exps` folder. Simply `cd` into this folder and run `bash <folder_name>/<script_name>.sh`. Experiments for Table 1
in the paper correspond to the folders `mnist_downstream`, `kmnist_downstream`, and `svhn32_downstream`. For Table 3, consult `svhn256_downstream`.

### Training the models

In order to launch experiments, we use the `task_launcher.py` script. This is a bit hefty at this point and contains a lot of argument options,
so it's recommended you get familiar with them by running `python task_launcher.py --help`. You can also see various examples of its usage by looking at the experimental scripts in the `exps` folder.


### Evaluating samples

This also easy! Simply add `--mode=interp_train` (or `--mode=interp_valid`) to the script. This changes the mode in the task launcher script
from training (which is the default) to interpolation mode. In this mode, interpolations between samples will be produced and output in the
results folder. The number of samples used for interpolation is dependent on `--val_batch_size`.

## Notes

- The main architecture we use here is one derived from a PyTorch reimplementation of ACAI, courtesy of Kyle McDonald, whose implementation can be found here: https://gist.github.com/kylemcdonald/e8ca989584b3b0e6526c0a737ed412f0
  - The main changes we make is that we add spectral norm to the discriminator to stabilise GAN training. We also added instance norm to the generator to stabilise training.
  - Generator code: https://github.com/christopher-beckham/amr/blob/dev/architectures/arch_kyle.py#L21-L96
  - Discriminator code: https://github.com/christopher-beckham/amr/blob/dev/architectures/arch_kyle.py#L98-L108

## Troubleshooting

If you are experiencing any issues, please file a ticket in the Issues section.
