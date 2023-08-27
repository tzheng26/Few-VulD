#!/bin/bash

# command lines to run the vulnerability detection system.

# To seperate the dataset in /Datasets into train, validation, and test data, run the following command.
if [ "$1" == "data" ]
then
    python main_meta.py --config config/config.yaml \
                        --dataset Datasets/SARD_4/processed_data \
                        --experiment_scheme Scheme_Array_API_SARD_4 \
                        --seed 26 \
                        --sep_train_vali_test \
                        --train_vali_test_dump train_vali_test

# For meta-training steps, please input the following code.
elif [ "$1" == "train" ]
then
    python main_meta.py --config config/config.yaml \
                        --train_or_test 0 \
                        --train_vali_test_load train_vali_test/train_vali_test7 \
                        --seed 27

# For meta-testting steps, please use the following commands:
elif [ "$1" == "test" ]
then
    python main_meta.py --config config/config.yaml \
                        --train_or_test 1 \
                        --train_vali_test_load train_vali_test/train_vali_test7 \
                        --trained_model Models/LSTM/2023-08-26-23-00-45/mamlMetaFormalLSTM25.h5 \
                        --seed 29

else
    echo "Error! Please run the script in one of the following formats."
    echo "./main_meta.sh Data"
    echo "or"
    echo "./main_meta.sh train"
    echo "or"
    echo "./main_meta.sh test"
fi
