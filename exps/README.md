## Folders

- For Table 1 (basic experiments, encoding dimension = 32)
  - `mnist`
  - `kmnist`
  - `svhn32`
- For Table 2 (ablation on SVHN, encoding dimension = 32)
  - `svhn32_20k` (20k train examples)
  - `svhn32_10k` (10k train examples)
  - `svhn32_5k` (5k train examples)
  - `svhn32_1k` (1k train examples)
- For Table 3 (SVHN + CIFAR10, larger encoding sizes)
  - `cifar10` (encoding dimension 256)
  - `cifar10_1024` (encoding dimension 1024)
  - `svhn256` (encoding dimension 256)
- For Table 4 (Dsprite experiments)
  - TODO
  
## Script names

Script names refer to the following:
- `baseline`: baseline AE
- `mixup`: mixup with k=2
- `mixupfm2`: Bernoulli mixup with k=2
- `ganXf`: mixup with k=X (e.g. `gan3f` = mix 3 examples at a time)
- `ganXfm`: Bernoulli mixup with k=X
- `acaif3`: ACAI
