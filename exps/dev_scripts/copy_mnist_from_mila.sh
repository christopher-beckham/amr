#!/bin/bash

# NOTE: need to run this locally with sshfs file systems
# cos some experiments are on cc and some are on mila.

mila_dir=~/Desktop/lisa_tmp4_4/github/swapgan_mixup/exps_mixup/s1
cc_dir=~/Desktop/beluga/github/swapgan_mixup/exps_mixup/s1

# Map from the (cryptic) experiment names in the private repo to the clean names
# in this public repo.
baseline=(10_mnist_baseline_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful.sh,mnist_baseline.sh)
mixup=(10_mnist_mixup_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful_sn.sh,mnist_mixup.sh)
bern=(10_mnist_mixupfm2_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful_sn.sh,mnist_bern.sh)
gan3f=(10_mnist_3gan-f_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful_sn.sh,mnist_mixup3.sh)
gan3fmf=(10_mnist_3ganfm-f_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful_sn.sh,mnist_bern3.sh)
acai_lamb2=(10_mnist_acaif_g8_frontal_bid_lamb2_inorm_32d_kyle_faithful_sn.sh,mnist_acai_lamb2.sh)
acai_mse_lamb2=(10_mnist_acaif-mse_g8_frontal_bid_lamb2_inorm_32d_kyle_faithful_sn.sh,mnist_acai-mse_lamb2.sh)

dst_dir=mnist_downstream
if [ ! -d $dst_dir ]; then
  mkdir $dst_dir
fi

for i in $baseline $mixup $bern $gan3f $gan3fmf $acai_lamb2 $acai_mse_lamb2; do IFS=",";
  set $i;
  src=$1
  echo "Processing $src ..."
  dst=$2
  if [ -e ${mila_dir}/mnist32_downstream_kyle/$src ]; then
    # Copy from mila server.
    cp ${mila_dir}/mnist32_downstream_kyle/$src $dst_dir/$dst
  else
    # Copy from CC server.
    cp ${cc_dir}/mnist32_downstream_kyle/$src $dst_dir/$dst
  fi
  # Quick script to replace NAME= in the bash script.
  dst_stripped="${dst:0:${#dst}-3}"
  python util/replace_line.py \
    $dst_dir/$dst \
    "NAME=" \
    "NAME=${dst_stripped}" > $dst_dir/$dst.fixed
  rm $dst_dir/$dst
  mv $dst_dir/$dst.fixed $dst_dir/$dst
done


