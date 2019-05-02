# Adversarial mixup resynthesizers

In this paper, we explore new approaches to combining information encoded within the learned representations of autoencoders. We explore models that are capable of combining the attributes of multiple inputs such that a resynthesised output is trained to fool an adversarial discriminator for real versus synthesised data. Furthermore, we explore the use of such an architecture in the context of semi-supervised learning, where we learn a mixing function whose objective is to produce interpolations of hidden states, or masked combinations of latent representations that are consistent with a conditioned class label. We show quantitative and qualitative evidence that such a formulation is an interesting avenue of research.

<img src="https://user-images.githubusercontent.com/2417792/57037002-ef8a8a80-6c23-11e9-9ecd-51582cd91258.png" width=768px />

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

### Training the models

```
python task_launcher.py \
..
..
..
```

To understand what each keyword argument is, simply run `python train.py --help`.

## Results

See paper.

| Dataset       |# hidden units| Interpolations|
| ------------- |:------------:|:-------------:|
| MNIST         |32            | ![image](https://user-images.githubusercontent.com/2417792/57098483-3641b880-6ce8-11e9-9b46-3dbb71f0094b.png) |

## Troubleshooting

If you are experiencing any issues, please file a ticket in the Issues section.
