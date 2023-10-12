# ***Few-VulD: A Few-shot Learning Method for Software Vulnerability Detection***

This is a meta-learning-based vulnerability detection system.

It is based on the meta-learning algorithm MAML.

Paper: [Few-shot-VulDetect](unavailable_yet)

Github: [Few-shot-VulDetect ](https://github.com/tzheng26/Few-VulD)

***The code will be fully open-sourced after the paper is accepted.***

## **1. Experiment Environment**

The experimental environment needs to be adjusted according to the computer hardware configuration.

Our experimental equipment configuration and environment are as follows:

|      Name      |            Function            |                     Version                     |
| :------------: | :----------------------------: | :---------------------------------------------: |
|      CPU      |    Central Processing Unit    | 13th Gen Intel(R) Core(TM) i7-13700KF @ 3.40GHz |
|      GPU      |    Graphics Processing Unit    |         NVIDIA GeForce RTX 4070 Ti 12G         |
|     Ubuntu     |        Operation System        |                   20.04.6 LTS                   |
|   Anaconda 3   | Python framework and libraries |                     23.5.2                     |
|     Python     |      Programming Language      |                     3.10.8                     |
|      CUDA      |         GPU Computing         |                     11.8.0                     |
|     cuDNN     |    GPU-accelerated library    |                    8.8.0.121                    |
| Tensorflow-gpu |    Deep Learning Framework    |                     2.10.0                     |
|     Keras     |    Deep Learning Framework    |                     2.10.0                     |
|     Numpy     |      Scientific Computing      |                     1.25.2                     |
|     Pandas     |         Data Analysis         |                      2.0.3                      |
|  Scikit-learn  |        Machine Learning        |                      1.3.0                      |
|     Gensim     |            Word2Vec            |                      4.3.1                      |
|   matplotlib   |         Visualization         |                      3.7.2                      |
|      yaml      |         Configuration         |                      0.2.5                      |

## **2. Installation**

We build the experiment environment with `conda`. Anaconda or mini-conda has to be installed.

Create conda environment:

```bash
conda create -n Few-VulD
conda activate Few-VulD
```

Install Tensorflow-gpu, CUDA and cuDNN according to your computer hardware setups.

Some versions of cuda and cudnn are not involved in conda's default install channel. Use the command below to add channel "conda-forge"

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Install tensorflow-gpu

```
conda install tensorflow-gpu=2.10.0
```

In our test, it automatically installed the following packages:

```bash
cudatoolkit=11.8.0 
cudnn=8.8.0.121
python=3.10.8
scipy=1.11.1
numpy=1.25.2
keras=2.10.0
keras-preprocessing=1.1.2
...
```

Install other needed packages:

```bash
conda install pyyaml pandas matplotlib scikit-learn tqdm gensim 
```

## **3. Quick Start**

```bash
# Split dataset into training, validation, and test sets.
./main_meta.sh data

# Start meta-training.
./main_meta.sh train

# Start meta-testing.
./main_meta.sh test

# Result will be stored in the "result_analysis" folder.
ls result_analysis
```

## **4. Datasets**

The system uses the SARD (Standard open-source vulneraebility dataset).

- Note: Customized dataset can also be integrated into the system.

**1. SARD**

- Directory: "Datasets/SARD_4".
- Program samples: "Datasets/SARD_4/Program_Samples".
- CWE information: "Datasets/SARD_4/Program_Samples/CWE_info/sard_info.json".
- *README.md* presents the description of the dataset.

**2. Customized Dataset**

- The system also supports user customized Dataset.
- Just make sure the dataset satisfies the required format of our system.

## **5. Data Preprocessing**

Data preprocessing includes removing code comments, removing empty lines, merging functions called across files, code slices, replacing program identifiers like user-defined variable names and function names, and distributing labels to each sample.

**For samples in SARD_4:** Go to the directory of the dataset

```bash
cd ./Datasets/SARD_4
```

Turn the original program files in folders `"API function call"`, `"Arithmetic expression"`, `"Array usage"`, and `"Pointer usage"` into seperated slice files and store them in the `"Program_Samples"` folder. A description file is stored as `"Program_Samples/CWE_info/sard_info.json"` to record the CWE types of each sample.

(*Due to the large amount of code snippets, we only use part of the data whose name contains "CWE" (i.e., the Juliet test suite).*)

```bash
python slices.py
```

Remove comments, empty lines, merge function calls across files, and replace user-defined variables with symbols.

```bash
python utils.py Program_Samples Pre_Program_Samples 
# Program_Samples is the location where all the program samples are stored; 
# Pre_Program_Samples is the directory to store preprocessed dataset.
```

The preprocessed dataset is stored in folder `Pre_Program_Samples`

## **6. Code Embedding**

**a) Pretrain the Word2Vec Model.**

Go to directory `./src/w2v`

```bash
cd src/w2v
```

Run `Word_to_vec_embedding.py` to train the **Word2Vec** model.

```bash
python Word_to_vec_embedding.py --data_dir	<Directory of the dataset>\
				--output_dir	<Directory of the output trained w2v model>\
				--n_workers	<Number of threads for training>\
				--size		<Dimensionality of the word vectors. This is the Embedding dimension.>\
				--window	<Maximum distance between the current and predicted word within a sentence>\
				--min_count	<Ignores all words with total frequency lower than this>\
				--algorithm	<Training algorithm: 1 for skip-gram; otherwise CBOW>\
				--seed		<Seed for the random number generator>\
```

For a quick start, the user could simply use the shell script `Word2vec.sh` with default settings:

```bash
./Word2vec.sh SARD_4
# or
./Word2vec.sh Custormized_Data # (Need to edit the shell script and change the value of parameter --data_dir to your own dataset's location.)
```

The result of the **w2v** model will be stored in folder `Few-VulD/w2v` by default.

**b) Transfer programs into embeddings.**

Run `Few-VulD/src/prep.py` with the following parameters to transfer programs to token sequences and labels:

```bash
python prep.py --Dataset   <Capitalized name of the dataset> \
              --data_path <PATH of custormized dataset> \
              --json_path <PATH of the JSON file> \
              --tokenizer_path <PATH to the tokenizer obtained from Word_to_vec_embedding.py> \
              --save_path <PATH to save the processed dataset>
```

(Note: --data_path, --tokenizer_path, and --json_path are required for custormized dataset.
The path to the tokenizer obtained from Word_to_vec_embedding.py is embedded in the script for system supported dataset. For user cusomized dataset, path to the tokenzier is needed.

For system supported datasets, run the script like:

```bash
python prep.py --Dataset SARD_4
```

Note: The above operations are only suitable for system supported datasets. Please identify all the required parameters for customized dataset.

`prep.py` transfers the dataset into the format below.  (Note: here is not the true embedding, its the numeric token sequeces of all program slices. Later, they will be transfered into matrices later in the system.)

```JSON
candi_Data{
    "CWE-1":{
        "Embeddings":[
            [1000-d numeric token sequence of file 0] (e.g., [ 17  26   1 ...   0   0   0]),
            ...,
            [1000-d numeric token sequence of file m-1] (e.g., [ 17 146   1 ...   0   0   0]),
            ],
        "Labels":[1, ..., 1]
    },
    "CWE-2":{
        "Embeddings":[
            [1000-d token sequence of file 0],
            ...,
            [1000-d token sequence of file m-1],
            ],
        "Labels":[1, ..., 1]
    },
    ...,
    "Benign":{
        "Embeddings":[
            [1000-d token sequence of file 0],
            ...,
            [1000-d token sequence of file n-1],
            ],
        "Labels":[0, ..., 0]
    }
}
```

so that it can be further processed by the system.
**Note:
a) Here is not the true embedding, its the numeric token sequeces of all program slices. Later, they will be transfered into matrices in the system.
b)The last key of the dictionary should be "Benign".**

The processed dataset will be dumped into <save_path>. User can use the `read_pickle.py` to help understand the basic structure of stored processed dataset.

For SARD_4, the structure of candi_data is:

```bash
{'API function call' : {
	'CWE-1': { 
		'Embeddings' : [
			[numeric tokens 1], 
			..., 
			[numeric tokens 1000]
			],
		'Labels' : [1, ..., 1]
		},
		...,
	'Benign':{ 
		'Embeddings' : [
			[numeric tokens 1], 
			..., 
			[numeric tokens 1000]
			],
		'Labels' : [0, ..., 0]
		},
	},
 'Arithmetic expression':{
	'CWE-1': { 
		'Embeddings' : [
			[numeric tokens 1], 
			..., 
			[numeric tokens 1000]
			],
		'Labels' : [1, ..., 1]
		},
		...,
	'Benign':{ 
		'Embeddings' : [
			[numeric tokens 1], 
			..., 
			[numeric tokens 1000]
			],
		'Labels' : [0, ..., 0]
		},
	},
 'Array usage':{
	'CWE-1': { 
		'Embeddings' : [
			[numeric tokens 1], 
			..., 
			[numeric tokens 1000]
			],
		'Labels' : [1, ..., 1]
		},
		...,
	'Benign':{ 
		'Embeddings' : [
			[numeric tokens 1], 
			..., 
			[numeric tokens 1000]
			],
		'Labels' : [0, ..., 0]
		},
	},
 'Pointer usage':{
	'CWE-1': { 
		'Embeddings' : [
			[numeric tokens 1], 
			..., 
			[numeric tokens 1000]
			],
		'Labels' : [1, ..., 1]
		},
		...,
	'Benign':{ 
		'Embeddings' : [
			[numeric tokens 1], 
			..., 
			[numeric tokens 1000]
			],
		'Labels' : [0, ..., 0]
		},
	},
}
```

Here is an example:

```bash
candi_data is stored in a format of <class 'dict'>
The first layer of keys: dict_keys(['API function call', 'Arithmetic expression', 'Array usage', 'Pointer usage'])

Taking the first key API function call as an example, the second layer of keys for it are:
['CWE-590', 'CWE-690', 'CWE-617', 'CWE-319', 'CWE-126', 'CWE-36', 'CWE-122', 'CWE-23', 'CWE-762', 'CWE-121', 'CWE-789', 'CWE-194', 'CWE-457', 'CWE-321', 'CWE-427', 'CWE-195', 'CWE-114', 'CWE-127', 'CWE-134', 'CWE-259', 'CWE-256', 'CWE-197', 'CWE-400', 'CWE-253', 'CWE-124', 'CWE-416', 'CWE-761', 'CWE-369', 'CWE-415', 'CWE-78', 'CWE-680', 'CWE-773', 'CWE-90', 'CWE-675', 'CWE-401', 'CWE-506', 'CWE-666', 'CWE-535', 'CWE-606', 'CWE-325', 'CWE-510', 'CWE-244', 'CWE-758', 'CWE-190', 'CWE-191', 'CWE-252', 'CWE-688', 'CWE-223', 'CWE-467', 'CWE-364', 'CWE-591', 'CWE-123', 'CWE-681', 'CWE-404', 'CWE-15', 'CWE-475', 'CWE-222', 'CWE-338', 'CWE-426', 'CWE-242', 'CWE-685', 'CWE-534', 'CWE-780', 'CWE-571', 'CWE-476', 'CWE-464', 'CWE-469', 'CWE-665', 'CWE-196', 'CWE-526', 'CWE-605', 'CWE-398', 'CWE-367', 'CWE-563', 'CWE-775', 'Benign']

For the first cwe type in it, it has 207 samples.
The numeric tokens are:
[[ 17  26   1 ...   0   0   0]
 [ 17 146   1 ...   0   0   0]
 [ 17  26   1 ...   0   0   0]
 ...
 [ 17  26   1 ...   0   0   0]
 [ 17  26   1 ...   0   0   0]
 [ 17  26   1 ...   0   0   0]]
The labels are:
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

## **7. Setup the System -- Configuration File**

Before running the system, users should set the configuration file `config/config.yaml` to specify the parameters of the system.

```yaml
# config.yaml
# E.g.,
...
model: "BiLSTM"
...
tokenizer_path: "w2v/SARD/tokenizer.pickle"
embedding_model_path: "w2v/SARD/w2v_model.txt"
...
```

**Note: The path of pre-trained tokenized and w2v_model has to be correct.**

## **8. Split Training, Validation, and Test Data with Specific Experiment Scheme**

**Run `main.py` to generate training, validation, and test data.**

**Firstly**, users should specify the **experiment scheme**.

- Users could specify the system supported experiment scheme with the *'--experiment_scheme'* parameter
- Or insert customized experiment scheme to the *`helper_for_metaL.py`*.

**Then**, with specified dataset, users should seperate the dataset into **training, validation, and test data** using the `main_meta.py` script with the following parameters:

```bash
python main.py --config <Directory of the configuration file (Default: config/config.yaml).>
                    --dataset <Directory of the user specified dataset (E.g., Datasets/Data_six/processed_data).>
                    --experiment_scheme <Experiment scheme, which will affect the way to seperate training, validation, and test data (E.g., Random_test).>
                    --seed <Specify the random seed to make experiment reproducible (E.g., 445)>
                    --sep_train_vali_test <This parameter tells the script to seperate user specified dataset into training, validation, and testing data; No need to assign any value to this parameter.>
                    --train_vali_test_dump <The destination directory for spliting training, validation, and test data; E.g.,train_vali_test>
```

E.g.,

```bash
python main.py --config config/config.yaml \
                    --dataset Datasets/SARD_4/processed_data \
                    --experiment_scheme Scheme_Random_SARD_4 \
                    --sep_train_vali_test \
                    --train_vali_test_dump train_vali_test \
					--seed 445
```

The processed data will be dumped to the directory specified by the `--train_vali_test_dump` parameter, e.g., the folder **train_vali_test/train_vali_testN, where N inditicate N-th sets.**

## **9. Start Meta-training**

With splited training, validation, and test sets, run the `main_meta.py` script with the following parameters to start meta-training.

```bash
python main.py --config <Directory of the configuration file (Default: config/config.yaml).>
                    --train_or_test 0 
                    --train_vali_test_load <Directory of the loaded training, validation, and test dataset; E.g.,: train_vali_test/train_vali_test0>
                    --seed <Specify the random seed to make experiment reproducible (E.g., 445)>
```

## **10. Start Meta-testing**

With splited training, validation, and test sets, run the `main_meta.py` script with the following parameters to start meta-testing.

```bash
python main.py --config <Directory of the configuration file (Default: config/config.yaml).>
                    --train_or_test 1 
                    --train_vali_test_load <Directory of the loaded training, validation, and test dataset; E.g.,: train_vali_test/train_vali_test0>
                    --trained_model <Path to the trained model> 
                    --seed <Specify the random seed to make experiment reproducible (E.g., 445)>
```

## **Contributors**

- Tianming Zheng
- Haojun Liu
- Hang Xu
- Xiang Chen
- Xinhao Li
- Ping Yi *
- Yue Wu
