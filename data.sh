#!/bin/bash

DATA_DIR=data
DS_NAME=QB
DS_ZIP_ID=1dLG5j-UEvGb8jUGcnSXV2opyoY0X9Tb1
DS_ZIP_NAME="$DS_NAME.zip"

if [ -d $DATA_DIR ]; then
  echo $DATA_DIR "folder exists"
else
  mkdir $DATA_DIR
fi

cd $DATA_DIR

source activate ceal # remove this line to run in different virtual env

gdown --id $DS_ZIP_ID
unzip -o -q $DS_ZIP_NAME

cd ../src
python data2.py