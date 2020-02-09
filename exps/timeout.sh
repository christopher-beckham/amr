#!/bin/bash


HOW_LONG=10 # how many seconds to wait

for folder in `find . -type d | tail -n +2`; do
  echo "Processing: " $folder
  for script in `ls $folder/*.sh`; do
    timeout --preserve-status $HOW_LONG bash $script
    echo "Exit code for $script:" $?
  done
done

# 143 = timeout
# 0 = good!
# else = bad
