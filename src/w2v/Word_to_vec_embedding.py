# -*- coding: utf-8 -*-
# Train a Word2vec model on the code base (unsupervised).
import argparse
import os
import time
import pickle
from gensim.models import Word2Vec
from tensorflow.compat.v1.keras.preprocessing.text import Tokenizer

# from src.prep import get_all_files_and_ids
import sys

sys.path.append("..")
from prep import get_all_files_and_ids


def log_file_name():
    """生成日志文件名"""
    log_file_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.txt'
    return log_file_name


if __name__ == '__main__':
    # 将实验结果输出到文件中
    log_filename = log_file_name()
    log_path = "../../w2v/logs"
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    if os.path.isfile(log_path + os.sep + log_filename):
        log_filename = (
            os.path.splitext(log_filename)[0]
            + "_new"
            + os.path.splitext(log_filename)[1]
        )
    log_path = log_path + os.sep + log_filename
    LogOutputFile = open(log_path, 'w')
    sys.stdout = LogOutputFile
    sys.stderr = LogOutputFile

    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a Word2vec model on the code base.'
    )
    parser.add_argument(
        '--data_dir',
        default='Datasets/',
        type=str,
        help='The path of the code base for training.',
    )
    parser.add_argument(
        '--output_dir',
        default='../../w2v/',
        type=str,
        help='The output path of the trained Word2vec model.',
    )
    parser.add_argument(
        '--n_workers',
        default=4,
        type=int,
        help='Number of threads for training.',
        required=False,
    )
    parser.add_argument(
        '--size',
        default=100,
        type=int,
        help='Dimensionality of the word vectors. This is the Embedding dimension.',
        required=False,
    )
    parser.add_argument(
        '--window',
        default=5,
        type=int,
        help='Maximum distance between the current and predicted word within a sentence.',
        required=False,
    )
    parser.add_argument(
        '--min_count',
        default=5,
        type=int,
        help='Ignores all words with total frequency lower than this.',
        required=False,
    )
    parser.add_argument(
        '--algorithm',
        default=0,
        type=int,
        help='Training algorithm: 1 for skip-gram; otherwise CBOW.',
        required=False,
    )
    parser.add_argument(
        '--seed',
        default=1,
        type=int,
        help='Seed for the random number generator.',
        required=False,
    )
    paras = parser.parse_args()

    # --------------------------------------------------------#
    # Idenfity the dataset
    Dataset = ""
    if "Data_six" in paras.data_dir:
        Dataset = "Data_six"
    elif "SARD_4" in paras.data_dir:
        Dataset = "SARD_4"
    elif "SARD" in paras.data_dir and "SARD_4" not in paras.data_dir:
        Dataset = "SARD"
    else:
        Dataset = "Customized_Data"

    print(
        "Now is pretraining the word2vec model based on the dataset: " + Dataset + "."
    )

    # Check the path
    print("The assigned path of dataset is: " + paras.data_dir)
    if not os.path.exists(paras.data_dir):
        print("The input path of dataset does not exist.")
        exit(0)
    if not os.path.exists(paras.output_dir):
        os.makedirs(paras.output_dir)
    print(
        "The output path of the trained Word2vec model is: "
        + os.path.join(paras.output_dir, Dataset)
    )

    # Identify files
    # total_list:		the token sequence of all text in the .c, .cpp files from paras.data_dir; a list.
    # total_list_id:	the name of all .c, .cpp files in paras.data_dir.
    total_list, total_list_id = get_all_files_and_ids(paras.data_dir)
    print("The amount of c or cpp files is : " + str(len(total_list_id)))
    print("top 10 total_list: ", total_list[0:10])  # TODO: top 10 是不是改成 the firt 10?
    print()
    print(
        "top 10 total_list_id: ", total_list_id[0:10]
    )  # TODO: top 10 是不是改成 the firt 10?

    # --------------------------------------------------------#
    # 1. Tokenization: convert the loaded text
    tokenizer = Tokenizer(
        num_words=None, filters=',', lower=False, char_level=False, oov_token=None
    )
    tokenizer.fit_on_texts(total_list)

    # Save the tokenizer.
    tokenizer_path = os.path.join(paras.output_dir, Dataset)
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)
    tokenizer_file_path = os.path.join(paras.output_dir, Dataset, 'tokenizer.pickle')
    with open(tokenizer_file_path, 'wb') as handle:
        pickle.dump(tokenizer, handle)

    # ----------------------------------------------------- #
    # 2. Train a Vocabulary with Word2Vec -- using the function provided by gensim.
    w2vModel = Word2Vec(
        total_list,
        workers=paras.n_workers,
        size=paras.size,
        window=paras.window,
        min_count=paras.min_count,
        sg=paras.algorithm,
        seed=paras.seed,
    )

    print("----------------------------------------")
    print("The trained word2vec model: ")
    print(w2vModel)

    w2v_path = os.path.join(paras.output_dir, Dataset, 'w2v_model.txt')
    w2vModel.wv.save_word2vec_format(w2v_path, binary=False)

    print()
    print()

    # # 加了个if语句
    # if paras.algorithm == 1:
    #     w2vModel.wv.save_word2vec_format(
    #         paras.output_dir + "w2v_model_skip-gram_dict.txt", binary=False
    #     )
    # else:
    #     w2vModel.wv.save_word2vec_format(
    #         paras.output_dir + "w2v_model_CBOW_dict.txt", binary=False
    #     )

    LogOutputFile.close()
