#!/bin/bash

# command lines to run the vulnerability detection system.

# To seperate the dataset in /Datasets into train, validation, and test data, run the following command.
if [ "$1" == "data" ]
then
    python main_meta.py --config config/config.yaml --dataset Datasets/SARD_4/processed_data --experiment_scheme Scheme_Random_SARD_4 --seed 445 --sep_train_vali_test --train_vali_test_dump train_vali_test

# For meta-training steps, please input the following code.
elif [ "$1" == "train" ]
then
    python main_meta.py --config config/config.yaml --train_or_test 0 --train_vali_test_load train_vali_test/train_vali_test15 --seed 26

# For meta-testting steps, please use the following commands:
elif [ "$1" == "test" ]
then
    python main_meta.py --config config/config.yaml --train_or_test 1 --train_vali_test_load train_vali_test/train_vali_test15 --trained_model Models/LSTM/2023-08-03-14-28-50/mamlMetaFormalLSTM35.h5 --seed 26

else
    echo "Error! Please run the script in one of the following formats."
    echo "./main_meta.sh Data"
    echo "or"
    echo "./main_meta.sh train"
    echo "or"
    echo "./main_meta.sh test"
fi
