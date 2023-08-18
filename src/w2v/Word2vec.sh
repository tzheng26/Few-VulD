#!/bin/bash
if [ "$1" == "Data_six" ]
then
    python Word_to_vec_embedding.py --data_dir ../../Datasets/Data_six/Pre_Six_project/symbols

elif [ "$1" == "SARD" ]
then
    python Word_to_vec_embedding.py --data_dir ../../Datasets/SARD/Pre_SARD/SARD_samples

elif [ "$1" == "SARD_4" ]
then
    python Word_to_vec_embedding.py --data_dir ../../Datasets/SARD_4/Pre_Program_Samples/symbols

elif [ "$1" == "Custormized_Data" ]
then
    python Word_to_vec_embedding.py --data_dir ../../Datasets/Custormized_Data/Custormized_Data

else
    echo "Error! Please run the script in one of the following formats."
    echo "./Word2vec.sh Data_six"
    echo "or"
    echo "./Word2vec.sh SARD"
    echo "or"
    echo "./Word2vec.sh SARD_4"
    echo "or"
    echo "./main_meta.sh Custormized_Data"
fi