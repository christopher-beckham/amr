#!/bin/bash

# Map from the (cryptic) experiment names in the private repo to the clean names
# in this public repo.
baseline=(15_svhn256_baseline_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful_sn.sh,svhn256_baseline.sh)
mixup=(15_svhn256_mixup_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful_sn.sh,svhn256_mixup.sh)
bern=(15_svhn256_mixupfm2_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful_sn.sh,svhn256_bern.sh)
gan3f=(15_svhn256_3gan-f_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful_sn.sh,svhn256_mixup3.sh)
gan3fmf=(15_svhn256_3ganfm-f_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful_sn.sh,svhn256_bern3.sh)
acai_lamb2=(15_svhn256_acaif_g8_frontal_bid_lamb2_inorm_32d_kyle_faithful_sn.sh,svhn256_acai_lamb2.sh)
acai_mse_lamb2=(15_svhn256_acaif-mse_g8_frontal_bid_lamb2_inorm_32d_kyle_faithful_sn.sh,svhn256_acai-mse_lamb2.sh)

dst_dir=svhn256_downstream
if [ ! -d $dst_dir ]; then
  mkdir $dst_dir
fi

for i in $baseline $mixup $bern $gan3f $gan3fmf $acai_lamb2 $acai_mse_lamb2; do IFS=",";
  set $i;
  src=$1
  echo "Processing $src ..."
  dst=$2
  cp ../../swapgan_mixup/exps_mixup/s1/svhn256_downstream_kyle/$src $dst_dir/$dst
  #sed -i "/NAME/c\NAME=${dst}" $dst_dir/$dst
  # Quick script to replace NAME= in the bash script.
  dst_stripped=${dst::-3}
  python util/replace_line.py \
    $dst_dir/$dst \
    "NAME=" \
    "NAME=${dst_stripped}" > $dst_dir/$dst.fixed
  rm $dst_dir/$dst
  mv $dst_dir/$dst.fixed $dst_dir/$dst
done


