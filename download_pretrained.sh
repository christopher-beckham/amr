#!/bin/bash

MNIST_ID=1OU_gT1Abp-7OrPtMjVChP-u2gjVxjHio

if [ ! -d results ]; then
  mkdir results
fi

cd results
echo 'Downloading from id ' ${MNIST_ID} '...'
gdown https://drive.google.com/uc?id=${MNIST_ID} -O mnist.zip
unzip mnist.zip
rm mnist.zip
