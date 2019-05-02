#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --gres=gpu
#SBATCH --mem=12G

# This is a convenience script which can be used to
# launch multiple experiments at once, e.g.
# ./launch_any.sh script1.sh script2.sh ...
# If you are using sbatch, you can simply change
# the sbatch variables above to what you desire.

##########################
# source activate amr ####
##########################
source activate pytorch-env

for script in "$@"
do
    echo "Launching $script ..."
    ./${script} &
done

wait
