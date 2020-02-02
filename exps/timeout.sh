#!/bin/bash


HOW_LONG=15 # how many seconds to wait

for folder in cifar10  cifar10_1024  kmnist  mnist  svhn256  svhn32  svhn32_10k  svhn32_1k  svhn32_20k  svhn32_5k; do
  for script in `ls $folder/*.sh`; do
    timeout --preserve-status $HOW_LONG bash $script
    echo "Exit code for $script:" $?
  done
done

# 143 = timeout
# 0 = good!
# else = bad
