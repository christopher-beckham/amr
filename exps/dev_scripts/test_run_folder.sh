#!/bin/bash

FOLDER=$1
DURATION=25s
if [ -z $FOLDER ]; then
  echo "Usage: test_run_folder.sh <folder_name>"
else
  for filename in `ls $FOLDER`; do
    echo "Running $filename for $DURATION ..."
    timeout ${DURATION} bash ${FOLDER}/$filename
  done
fi
