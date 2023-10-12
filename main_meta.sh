#!/bin/bash

# command lines to run the vulnerability detection system.

# To seperate the dataset in /Datasets into train, validation, and test data, run the following command.
if [ "$1" == "data" ]
then
    python main.py  --config config/config.yaml \
                    --dataset Datasets/SARD_4/processed_data \
                    --experiment_scheme Scheme_Array_API_SARD_4 \
                    --seed 26 \
                    --sep_train_vali_test \
                    --train_vali_test_dump train_vali_test

# For meta-training steps, please input the following code.
elif [ "$1" == "train" ]
then
    python main.py  --config config/config.yaml \
                    --train_or_test 0 \
                    --train_vali_test_load train_vali_test/train_vali_test7 \
                    --seed 27

# For meta-testting steps, please use the following commands:
elif [ "$1" == "test" ]
then
    python main.py  --config config/config.yaml \
                    --train_or_test 1 \
                    --train_vali_test_load train_vali_test/train_vali_test50 \
                    --trained_model Models/BiLSTM/2023-09-17-01-45-51/mamlMetaFormalBiLSTM46.h5 \
                    --seed 924

elif [ "$1" == "automode" ]
then
    python main.py  --config config/config.yaml \
                    --dataset Datasets/SARD_4/processed_data \
                    --experiment_scheme Scheme_Array_SARD_4 \
                    --train_vali_test_dump train_vali_test \
                    --automode 1
                    
else
    echo "Error! Please run the script in one of the following formats."
    echo "./main.sh Data"
    echo "or"
    echo "./main.sh train"
    echo "or"
    echo "./main.sh test"
    echo "or"
    echo "./main.sh automode"
fi
