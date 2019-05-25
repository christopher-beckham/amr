#!/bin/bash

# Map from the (cryptic) experiment names in the private repo to the clean names
# in this public repo.
baseline=(15_svhn_baseline_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful.sh,svhn32_baseline.sh)
mixup=(15_svhn_mixup_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful_sn.sh,svhn32_mixup.sh)
bern=(15_svhn_mixupfm2_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful_sn.sh,svhn32_bern.sh)
gan3f=(15_svhn_3gan-f_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful_sn.sh,svhn32_mixup3.sh)
gan3fmf=(15_svhn_3ganfm-f_g8_frontal_bid_lamb10_inorm_32d_kyle_faithful_sn.sh,svhn32_bern3.sh)
acai_lamb2=(15_svhn_acaif_g8_frontal_bid_lamb2_inorm_32d_kyle_faithful_sn.sh,svhn32_acai_lamb2.sh)
acai_mse_lamb2=(15_svhn_acaif-mse_g8_frontal_bid_lamb2_inorm_32d_kyle_faithful_sn.sh,svhn32_acai-mse_lamb2.sh)

dst_dir=svhn32_downstream
if [ ! -d $dst_dir ]; then
  mkdir $dst_dir
fi

for i in $baseline $mixup $bern $gan3f $gan3fmf $acai_lamb2 $acai_mse_lamb2; do IFS=",";
  set $i;
  src=$1
  echo "Processing $src ..."
  dst=$2
  cp ../../swapgan_mixup/exps_mixup/s1/svhn32_downstream_kyle/$src $dst_dir/$dst
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


