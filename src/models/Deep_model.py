# -*- coding: utf-8 -*-

from tensorflow.compat.v1.keras import losses
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import (
    Input,
    Dense,
    Embedding,
    Flatten,
    Bidirectional,
    CuDNNGRU,
    CuDNNLSTM,
    GlobalMaxPooling1D,
    LSTM,
    Dropout,
)

# from tensorflow.compat.v1.keras.layers.core import
import tensorflow as tf
from sklearn.utils import class_weight
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import datetime


class Deep_model(object):
    def __init__(self, config, word_index, embedding_matrix):
        self.MAX_LEN = config['model_settings']['model_para']['max_sequence_length']
        self.EMBEDDING_DIM = config['model_settings']['model_para']['embedding_dim']
        self.use_dropout = config['model_settings']['model_para']['use_dropout']
        self.dropout_rate = config['model_settings']['model_para']['dropout_rate']
        self.LOSS_FUNCTION = config['model_settings']['loss_function']
        self.OPTIMIZER = config['model_settings']['optimizer']['type']
        self.dnn_size = config['model_settings']['model_para']['dnn_size']
        self.rnn_size = config['model_settings']['model_para']['rnn_size']
        self.embedding_trainable = config['model_settings']['model_para'][
            'embedding_trainable'
        ]

        # Load the model.
        print("-------------------------------------------------------")
        if config['model_settings']['model'] == 'DNN':
            print("Loading the " + config['model_settings']['model'] + " model.")
            self.meta_model = self.build_DNN(word_index, embedding_matrix)
        if config['model_settings']['model'] == 'GRU':
            print("Loading the " + config['model_settings']['model'] + " model.")
            self.meta_model = self.build_GRU(word_index, embedding_matrix)
        if config['model_settings']['model'] == 'LSTM':
            print("Loading the " + config['model_settings']['model'] + " model.")
            self.meta_model = self.build_LSTM(word_index, embedding_matrix)
        if config['model_settings']['model'] == 'BiGRU':
            print("Loading the " + config['model_settings']['model'] + " model.")
            self.meta_model = self.build_BiGRU(word_index, embedding_matrix)
        if config['model_settings']['model'] == 'BiLSTM':
            print("Loading the " + config['model_settings']['model'] + " model.")
            self.meta_model = self.build_BiLSTM(word_index, embedding_matrix)
        # if config['model_settings']['model'] == 'textCNN':
        #    print ("Loading the " + self.config['model_settings']['model'] + " model.")
        #    self.model = test_CNN.buildModel(word_index, embedding_matrix)

    # 基于keras定义各个模型
    def build_DNN(self, word_index, embedding_matrix):
        inputs = Input(shape=(self.MAX_LEN,))
        sharable_embedding = Embedding(
            len(word_index) + 1,
            self.EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=self.MAX_LEN,
            trainable=self.embedding_trainable,
        )(inputs)
        dense = Flatten()(sharable_embedding)
        dense_0 = Dense(self.dnn_size, activation='relu')(dense)

        if self.use_dropout:
            dropout_layer_2 = Dropout(self.dropout_rate)(dense_0)
            dense_1 = Dense(self.dnn_size, activation='relu')(dropout_layer_2)
        else:
            dense_1 = Dense(self.dnn_size, activation='relu')(dense_0)

        if self.use_dropout:
            dropout_layer_3 = Dropout(self.dropout_rate)(dense_1)
            dense_2 = Dense(int(self.dnn_size / 2), activation='relu')(dropout_layer_3)
        else:
            dense_2 = Dense(int(self.dnn_size / 2), activation='relu')(dense_1)

        dense_3 = Dense(int(self.dnn_size / 4))(dense_2)
        dense_4 = Dense(1, activation='sigmoid')(dense_3)

        model = Model(inputs=inputs, outputs=dense_4, name='DNN_network')

        model.compile(
            loss=self.LOSS_FUNCTION, optimizer=self.OPTIMIZER, metrics=['accuracy']
        )

        return model

    def build_GRU(self, word_index, embedding_matrix):
        inputs = Input(shape=(self.MAX_LEN,))
        self.use_dropout = False
        sharable_embedding = Embedding(
            len(word_index) + 1,
            self.EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=self.MAX_LEN,
            trainable=self.embedding_trainable,
        )(inputs)
        gru_1 = CuDNNGRU(self.rnn_size, return_sequences=True)(
            sharable_embedding
        )  # The default activation is 'tanh',
        if self.use_dropout:
            droput_layer_1 = Dropout(self.dropout_rate)(gru_1)
            gru_2 = CuDNNGRU(self.rnn_size, return_sequences=True)(droput_layer_1)
        else:
            gru_2 = CuDNNGRU(self.rnn_size, return_sequences=True)(gru_1)

        gmp_layer = GlobalMaxPooling1D()(gru_2)

        if self.use_dropout:
            dropout_layer_2 = Dropout(self.dropout_rate)(gmp_layer)
            dense_1 = Dense(int(self.dnn_size / 2), activation='relu')(dropout_layer_2)
        else:
            dense_1 = Dense(int(self.dnn_size / 2), activation='relu')(gmp_layer)

        dense_2 = Dense(int(self.rnn_size / 4))(dense_1)
        dense_3 = Dense(1, activation='sigmoid')(dense_2)

        model = Model(inputs=inputs, outputs=dense_3, name='GRU_network')

        model.compile(
            loss=self.LOSS_FUNCTION, optimizer=self.OPTIMIZER, metrics=['accuracy']
        )

        return model

    def build_LSTM(self, word_index, embedding_matrix):
        # self.use_dropout = False
        inputs = Input(shape=(self.MAX_LEN,))
        sharable_embedding = Embedding(
            len(word_index) + 1,
            self.EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=self.MAX_LEN,
            trainable=self.embedding_trainable,
        )(inputs)
        # print("Shape\n\n\n\n\n",sharable_embedding.shape)
        gru_1 = CuDNNLSTM(self.rnn_size, return_sequences=True)(
            sharable_embedding
        )  # The default activation is 'tanh',

        if self.use_dropout:
            droput_layer_1 = Dropout(self.dropout_rate)(gru_1)
            gru_2 = CuDNNLSTM(self.rnn_size, return_sequences=True)(droput_layer_1)
        else:
            gru_2 = CuDNNLSTM(self.rnn_size, return_sequences=True)(gru_1)
        gmp_layer = GlobalMaxPooling1D()(gru_2)
        if self.use_dropout:
            dropout_layer_2 = Dropout(self.dropout_rate)(gmp_layer)
            dense_1 = Dense(int(self.dnn_size / 2), activation='relu')(dropout_layer_2)
        else:
            dense_1 = Dense(int(self.dnn_size / 2), activation='relu')(gmp_layer)
        dense_2 = Dense(int(self.rnn_size / 4))(dense_1)
        dense_3 = Dense(1, activation='sigmoid')(dense_2)
        model = Model(inputs=inputs, outputs=dense_3, name='LSTM_network')
        model.compile(
            loss=self.LOSS_FUNCTION, optimizer=self.OPTIMIZER, metrics=['accuracy']
        )

        return model

    def build_LSTM_New(self, word_index, embedding_matrix):
        self.use_dropout = False
        inputs = Input(shape=(self.MAX_LEN,))
        sharable_embedding = Embedding(
            len(word_index) + 1,
            self.EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=self.MAX_LEN,
            trainable=self.embedding_trainable,
        )(inputs)
        # gru_1 = CuDNNLSTM(self.rnn_size, return_sequences=True)(sharable_embedding) # The default activation is 'tanh',
        gru_1 = LSTM(self.rnn_size, return_sequences=True)(sharable_embedding)
        if self.use_dropout:
            droput_layer_1 = Dropout(self.dropout_rate)(gru_1)
            # gru_2 = CuDNNLSTM(self.rnn_size, return_sequences=True)(droput_layer_1)
            gru_2 = LSTM(self.rnn_size, return_sequences=True)(droput_layer_1)
        else:
            # gru_2 = CuDNNLSTM(self.rnn_size, return_sequences=True)(gru_1)
            gru_2 = LSTM(self.rnn_size, return_sequences=True)(gru_1)
        gmp_layer = GlobalMaxPooling1D()(gru_2)
        if self.use_dropout:
            dropout_layer_2 = Dropout(self.dropout_rate)(gmp_layer)
            dense_1 = Dense(int(self.dnn_size / 2), activation='relu')(dropout_layer_2)
        else:
            dense_1 = Dense(int(self.dnn_size / 2), activation='relu')(gmp_layer)
        dense_2 = Dense(int(self.rnn_size / 4))(dense_1)
        dense_3 = Dense(1, activation='sigmoid')(dense_2)
        model = Model(inputs=inputs, outputs=dense_3, name='LSTM_network')
        # model.compile(loss=self.LOSS_FUNCTION,
        #         optimizer=self.OPTIMIZER,
        #         metrics=['accuracy'])

        return model

    def build_BiGRU(self, word_index, embedding_matrix):
        # self.use_dropout = False
        inputs = Input(shape=(self.MAX_LEN,))

        sharable_embedding = Embedding(
            len(word_index) + 1,
            self.EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=self.MAX_LEN,
            trainable=self.embedding_trainable,
        )(inputs)
        bigru_1 = Bidirectional(
            CuDNNGRU(int(self.dnn_size / 2), return_sequences=True), merge_mode='concat'
        )(
            sharable_embedding
        )  # The default activation is 'tanh',
        if self.use_dropout:
            droput_layer_1 = Dropout(self.dropout_rate)(bigru_1)
            bigru_2 = Bidirectional(
                CuDNNGRU(int(self.dnn_size / 2), return_sequences=True),
                merge_mode='concat',
            )(droput_layer_1)
        else:
            bigru_2 = Bidirectional(
                CuDNNGRU(int(self.dnn_size / 2), return_sequences=True),
                merge_mode='concat',
            )(bigru_1)

        gmp_layer = GlobalMaxPooling1D()(bigru_2)

        if self.use_dropout:
            dropout_layer_2 = Dropout(self.dropout_rate)(gmp_layer)
            dense_1 = Dense(int(self.dnn_size / 2), activation='relu')(dropout_layer_2)
        else:
            dense_1 = Dense(int(self.dnn_size / 2), activation='relu')(gmp_layer)

        dense_2 = Dense(int(self.rnn_size / 4))(dense_1)
        dense_3 = Dense(1, activation='sigmoid')(dense_2)

        model = Model(inputs=inputs, outputs=dense_3, name='BiGRU_network')

        model.compile(
            loss=self.LOSS_FUNCTION, optimizer=self.OPTIMIZER, metrics=['accuracy']
        )

        return model

    def build_BiLSTM(self, word_index, embedding_matrix):
        # self.use_dropout = False
        inputs = Input(shape=(self.MAX_LEN,))
        self.use_dropout = False

        sharable_embedding = Embedding(
            len(word_index) + 1,
            self.EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=self.MAX_LEN,
            trainable=self.embedding_trainable,
        )(inputs)
        bilstm_1 = Bidirectional(
            CuDNNLSTM(int(self.dnn_size / 2), return_sequences=True),
            merge_mode='concat',
        )(
            sharable_embedding
        )  # The default activation is 'tanh',
        if self.use_dropout:
            droput_layer_1 = Dropout(self.dropout_rate)(bilstm_1)
            bilstm_2 = Bidirectional(
                CuDNNLSTM(int(self.dnn_size / 2), return_sequences=True),
                merge_mode='concat',
            )(droput_layer_1)
        else:
            bilstm_2 = Bidirectional(
                CuDNNLSTM(int(self.dnn_size / 2), return_sequences=True),
                merge_mode='concat',
            )(bilstm_1)

        gmp_layer = GlobalMaxPooling1D()(bilstm_2)

        if self.use_dropout:
            dropout_layer_2 = Dropout(self.dropout_rate)(gmp_layer)
            dense_1 = Dense(int(self.dnn_size / 2), activation='relu')(dropout_layer_2)
        else:
            dense_1 = Dense(int(self.dnn_size / 2), activation='relu')(gmp_layer)

        dense_2 = Dense(int(self.rnn_size / 4))(dense_1)
        dense_3 = Dense(1, activation='sigmoid')(dense_2)

        model = Model(inputs=inputs, outputs=dense_3, name='BiLSTM_network')

        model.compile(
            loss=self.LOSS_FUNCTION, optimizer=self.OPTIMIZER, metrics=['accuracy']
        )

        return model

    """
    Meta-training
    ------------------------------------------------------------------
    MAML一个batch的训练过程
    :param train_data: 训练数据，以task为一个单位
    :param inner_optimizer: support set对应的优化器
    :param inner_step: 内部更新几个step
    :param outer_optimizer: query set对应的优化器，如果对象不存在则不更新梯度
    :return: batch query loss
    """

    def train_on_metabatch(
        self,
        train_data,
        inner_optimizer,
        inner_step,
        outer_optimizer=None,
        losss=losses.BinaryCrossentropy(from_logits=False),
        class_weights=[1, 1],
    ):
        # print(self.meta_model.trainable_variables)
        batch_acc = []
        batch_pre = []
        batch_rec = []
        batch_TNR = []
        batch_FPR = []
        batch_FNR = []
        batch_F1_score = []
        batch_loss = []
        task_weights = []
        batch_Tp = 0
        batch_Tn = 0
        batch_Fp = 0
        batch_Fn = 0

        # 用meta_weights保存一开始的权重，并将其设置为inner step模型的权重
        meta_weights = self.meta_model.get_weights()
        # print("Weights: ", self.meta_model.get_weights(), '\n')
        # print(len(meta_weights))
        # for k in range(len(meta_weights)):
        #   print(len(meta_weights[k]))
        # return
        testW = meta_weights

        (
            meta_support_image,
            meta_support_label,
            meta_query_image,
            meta_query_label,
        ) = train_data
        # print("*****************************")
        # print("meta_support_label:\n", meta_support_label)
        # print()

        # print("-------------------")
        # print("Inner loop:\n")
        # inner loop: 一个 batch 中的每个任务
        for support_image, support_label in zip(meta_support_image, meta_support_label):
            # 每个task都需要载入最原始的weights进行更新，内层模型的初始参数为整个外层模型学习到的参数
            self.meta_model.set_weights(meta_weights)
            # print("support_image of a task:")
            # print(support_image, '\n')

            # print("support_label of a task:")
            # print(support_label, '\n')

            for _ in range(inner_step):
                with tf.GradientTape() as tape:
                    logits = self.meta_model(support_image, training=True)
                    # 在深度学习中，logits 是指神经网络输出层的未经激活的预测结果，也就是说，logits 是网络的输出，还没有经过 softmax 激活函数，其结果是一组实数值，代表网络对每个类别的预测结果，可以理解为对各个类别的打分。在这段代码中，logits 是使用元模型对支持图像进行预测的结果，它的形状为 (batch_size, num_classes)，其中 batch_size 是支持集中图像的数量，num_classes 是分类任务中类别的数量。后续的代码根据 logits 计算损失和准确率等评估指标。
                    # logits = self.meta_model.predict(support_image)
                    # print("logits:")
                    # print(logits, '\n')

                    # 转换成一列
                    support_label = support_label.reshape(-1, 1)
                    # print("1 colume support_label:")
                    # print(support_label, '\n')

                    # TODO handle class imbalance
                    sample_weights = []

                    ppred = []
                    for i in range(len(support_label)):
                        # TODO handle class imbalance
                        if support_label[i] == 1:
                            sample_weights.append(class_weights[1])
                        else:
                            sample_weights.append(class_weights[0])

                        if logits[i] < 0.5:
                            ppred.append(0)
                        else:
                            ppred.append(1)
                    # print("ppred:")
                    # print(ppred, '\n')

                    loss = losss(support_label, logits, sample_weight=sample_weights)
                    # print("loss")
                    # print(loss, '\n')
                    loss = tf.reduce_mean(loss)
                    # print("loss reduce_mean:")
                    # print(loss, '\n')

                    # print("tf.argmax:")
                    # print(tf.argmax(logits, axis=-1, output_type=tf.int32), '\n')
                    # 这块acc怎么计算的？？是否有用？？？
                    # TODO: 这块acc永远是0.5，这块明显不对
                    acc = tf.cast(
                        tf.argmax(logits, axis=-1, output_type=tf.int32)
                        == support_label,
                        tf.float32,
                    )
                    # print("acc:")
                    # print(acc, '\n')
                    acc = tf.reduce_mean(acc)
                    # print("acc reduce mean:")
                    # print(acc, '\n')

                grads = tape.gradient(loss, self.meta_model.trainable_variables)
                # grads = tf.clip_by_global_norm(grads)
                # print(grads[3])

                inner_optimizer.apply_gradients(
                    zip(grads, self.meta_model.trainable_variables)
                )

                # print("-------------------")

            # 每次经过inner loop更新过后的weights都需要保存一次，保证这个weights后面outer loop训练的是同一个task
            task_weights.append(self.meta_model.get_weights())
            # return

        # print("-------------------")
        # print("outer loop:")
        # outer loop:
        with tf.GradientTape() as tape:
            for i, (query_image, query_label) in enumerate(
                zip(meta_query_image, meta_query_label)
            ):
                # print('------------')
                # print("For Task", i, " in the batch:\n")

                # 载入每个task weights进行前向传播
                self.meta_model.set_weights(task_weights[i])

                # 真实标签
                query_label = query_label.reshape(-1, 1)
                # print("query label:")
                # print(query_label)
                # print()

                # 预测概率
                logits = self.meta_model(query_image, training=True)
                # print("logits:")
                # print(logits)
                # print()

                # TODO handle class imbalance
                sample_weights = []
                for i in range(len(query_label)):
                    if query_label[i] == 1:
                        sample_weights.append(class_weights[1])
                    else:
                        sample_weights.append(class_weights[0])

                # loss
                loss = losss(query_label, logits, sample_weight=sample_weights)
                loss = tf.reduce_mean(loss)
                # print("Loss:", loss)
                batch_loss.append(loss)

                ppred = []
                # recall = []
                # precision = []

                # 预测标签
                # shape[0]表示行数
                for i in range(logits.shape[0]):
                    num = 0
                    # print(logits[i][0])
                    if logits[i][0] > 0.5:
                        num = 1
                    ppred.append(num)
                    # if query_label[i][0] == 1:
                    #     # recall rate TP/(TP+FN)
                    #     recall.append(num)
                    # if num == 1:
                    #     # precision rate TP/(TP+FP)
                    #     precision.append(query_label[i][0])
                # print("ppred:")
                # print(ppred, '\n')

                # 计算accuracy，有问题，tf张量后面用np.mean计算平均值的时候有错误。
                # acc = tf.cast(ppred == query_label.flatten(), tf.float64)
                # acc = tf.reduce_mean(acc)
                # print("Accuracy:")
                # print(acc, '\n')

                ppred = np.array(ppred)
                ppred = ppred.flatten()
                query_label = query_label.flatten()
                # print("query_label.flatten():")
                # print(query_label, '\n')

                Tp = 0
                Fp = 0
                Tn = 0
                Fn = 0
                for label, pred in zip(query_label, ppred):
                    if (label == 1) and (pred == 1):
                        Tp = Tp + 1
                    elif (label == 1) and (pred == 0):
                        Fn = Fn + 1
                    elif (label == 0) and (pred == 0):
                        Tn = Tn + 1
                    elif (label == 0) and (pred == 1):
                        Fp = Fp + 1
                    else:
                        print('something weird with labels')
                        return -1
                # print("TP:", Tp)
                # print("FP:", Fp)
                # print("TN:", Tn)
                # print("FN:", Fn)
                # print()

                try:
                    acc = (Tp + Tn) / (Tp + Tn + Fp + Fn)
                except:
                    acc = 0
                try:
                    precision = Tp / (Tp + Fp)
                except:
                    precision = 0
                try:
                    recall = Tp / (Tp + Fn)
                except:
                    recall = 0
                try:
                    TNR = Tn / (Fp + Tn)
                except:
                    TNR = 0
                try:
                    FPR = Fp / (Fp + Tn)
                except:
                    FPR = 0
                try:
                    FNR = Fn / (Tp + Fn)
                except:
                    FNR = 0
                try:
                    F1_score = 2 * precision * recall / (precision + recall)
                except:
                    F1_score = 0
                # print("Metrics for a single task:")
                # print("Accuracy:", acc)
                # print("Precision:", precision)
                # print("Recall:", recall)
                # print("TNR:", TNR)
                # print("FPR:", FPR)
                # print("FNR", FNR)
                # print("F1-score:", F1_score)
                # print()

                # print("precision:")
                # print(precision, '\n')

                # print("recall:")
                # print(recall, '\n')

                batch_Tp = batch_Tp + Tp
                batch_Tn = batch_Tn + Tn
                batch_Fp = batch_Fp + Fp
                batch_Fn = batch_Fn + Fn
                # print("The whole batch's TP, FP, TN, FN:")
                # print("Batch TP:", batch_Tp)
                # print("Batch FP:", batch_Fp)
                # print("Batch TN:", batch_Tn)
                # print("Batch FN:", batch_Fn)
                # print()

                batch_acc.append(acc)
                batch_rec.append(recall)
                batch_pre.append(precision)
                batch_TNR.append(TNR)
                batch_FPR.append(FPR)
                batch_FNR.append(FNR)
                batch_F1_score.append(F1_score)
                # print("The whole batch's metrics through .append():")
                # print("Batch accuracy:", batch_acc)
                # print("Batch recall:", batch_rec)
                # print("Batch precision:", batch_pre)
                # print("Batch TNR:", batch_TNR)
                # print("Batch FPR:", batch_FPR)
                # print("Batch FNR:", batch_FNR)
                # print("Batch F1:", batch_F1_score)
                # print()

                # print("batch_acc")
                # print(batch_acc, '\n')

                # print("batch_rec")
                # print(batch_rec, '\n')

                # print("batch_pre")
                # print(batch_pre, '\n')

            mean_loss = tf.reduce_mean(batch_loss)
            # print("mean loss after tf.reduce_mean:", mean_loss)

            # mean_acc = tf.reduce_mean(batch_acc)
            mean_acc = np.mean(batch_acc)
            mean_rec = np.mean(batch_rec)
            mean_pre = np.mean(batch_pre)
            mean_TNR = np.mean(batch_TNR)
            mean_FPR = np.mean(batch_FPR)
            mean_FNR = np.mean(batch_FNR)
            mean_F1 = np.mean(batch_F1_score)
            # print(
            #     "Mean Batch Metrics through np.mean() of the appended batch metric list:"
            # )
            # print("mean_acc", mean_acc)
            # print("mean_rec", mean_rec)
            # print("mean_pre", mean_pre)
            # print("mean_TNR", mean_TNR)
            # print("mean_FPR", mean_FPR)
            # print("mean_FNR", mean_FNR)
            # print("mean_F1", mean_F1)
            # print()

            # # Wrong way to calculate the metrics.
            # print("Mean Batch Metrics through calculating the batch TP, TN, FP, FN:")
            # try:
            #     c_acc = (batch_Tp + batch_Tn) / (
            #         batch_Tp + batch_Tn + batch_Fp + batch_Fn
            #     )
            # except:
            #     c_acc = 0
            # try:
            #     c_precision = batch_Tp / (batch_Tp + batch_Fp)
            # except:
            #     c_precision = 0
            # try:
            #     c_recall = batch_Tp / (batch_Tp + batch_Fn)
            # except:
            #     c_recall = 0
            # try:
            #     c_TNR = batch_Tn / (batch_Fp + batch_Tn)
            # except:
            #     c_TNR = 0
            # try:
            #     c_FPR = batch_Fp / (batch_Fp + batch_Tn)
            # except:
            #     c_FPR = 0
            # try:
            #     c_FNR = batch_Fn / (batch_Tp + batch_Fn)
            # except:
            #     c_FNR = 0
            # try:
            #     c_F1_score = (
            #         2
            #         * batch_precision
            #         * batch_recall
            #         / (batch_precision + batch_recall)
            #     )
            # except:
            #     c_F1_score = 0
            # print("calculated acc:", c_acc)
            # print("calculated precision:", c_precision)
            # print("calculated recall:", c_recall)
            # print("calculated TNR:", c_TNR)
            # print("calculated FPR:", c_FPR)
            # print("calculated FNR:", c_FNR)
            # print("calculated F1:", c_F1_score)
            # print()

            # print("Check whether they are the same!!!!")

        # meta-training 阶段的训练集（包含sample set & query set），更新外层参数
        # meta-training 阶段的验证集（包含sample set & query set），不更新外层参数，只是看看当前模型参数对验证集的分类效果等。
        # 无论是否更新，都需要载入最开始的权重进行更新，防止val阶段改变了原本的权重
        self.meta_model.set_weights(meta_weights)
        if outer_optimizer:
            grads = tape.gradient(mean_loss, self.meta_model.trainable_variables)
            # for k in range(len(grads)):
            #  print(len(grads[k]))
            # for k in range(len(self.meta_model.trainable_variables)):
            #  print(self.meta_model.trainable_variables[k].shape)
            # return
            # print("grads: ",grads)
            outer_optimizer.apply_gradients(
                zip(grads, self.meta_model.trainable_variables)
            )

        return (
            mean_loss,
            mean_acc,
            mean_rec,
            mean_pre,
            mean_TNR,
            mean_FPR,
            mean_FNR,
            mean_F1,
        )

    """
    meta-testing
    ------------------------------------------
    针对某个cwe类型的测试：
    supportData = [supporttuple_0,supporttuple_1,....]
    supporttuple = [support_image, suport_label]
    supportData = [
        [
            [support_embed_1], 
            [support_label_1]
        ],
        [
            [support_embed_2], 
            [support_label_2]
        ]
    ]

    queryData = [querytuple_0, querytuple_1, querytuple_2, ...]
    querytuple = [query_image, query_label]

    现在需要一个 support image 的列表，包括support数据的所有embedding
    support label 列表，包含所有label

    """

    def test_on_one_type(
        self,
        support_image,
        support_label,
        query_image,
        query_label,
        inner_optimizer,
        inner_step=1,  # inner step 可以大一点
        losss=losses.BinaryCrossentropy(from_logits=False),
    ):
        # batch_acc = []
        # batch_pre = []
        # batch_rec = []
        # batch_loss = []
        # task_weights = []

        support_image = np.array(support_image)
        support_label = np.array(support_label)
        query_image = np.array(query_image)
        query_label = np.array(query_label)

        # 用meta_weights保存一开始的权重，并将其设置为inner step模型的权重
        meta_weights = self.meta_model.get_weights()
        self.meta_model.set_weights(meta_weights)

        # support 训练
        test_support_start_time = datetime.datetime.now()
        for _ in range(inner_step):
            with tf.GradientTape() as tape:
                logits = self.meta_model(support_image, training=True)
                # print(logits)
                support_label = support_label.reshape(-1, 1)
                ppred = []
                for i in range(len(support_label)):
                    if logits[i] < 0.5:
                        ppred.append(0)
                    else:
                        ppred.append(1)
                loss = losss(support_label, logits)
                # print(loss)
                loss = tf.reduce_mean(loss)
                # TODO:这块acc是否需要？这块不对！！！！
                # print(tf.argmax(logits, axis=-1, output_type=tf.int32))
                acc = tf.cast(
                    tf.argmax(logits, axis=-1, output_type=tf.int32) == support_label,
                    tf.float32,
                )
                acc = tf.reduce_mean(acc)
            grads = tape.gradient(loss, self.meta_model.trainable_variables)
            inner_optimizer.apply_gradients(
                zip(grads, self.meta_model.trainable_variables)
            )
        test_support_end_time = datetime.datetime.now()
        test_support_time = test_support_end_time - test_support_start_time

        # Query 测试
        test_query_start_time = datetime.datetime.now()

        logits = self.meta_model(query_image, training=True)
        query_label = query_label.reshape(-1, 1)
        loss = losss(query_label, logits)
        loss = tf.reduce_mean(loss)
        loss = loss.numpy()

        ppred = []
        # recall = []
        # precision = []
        for i in range(logits.shape[0]):
            num = 0
            # print(logits[i][0])
            if logits[i][0] > 0.5:
                num = 1
            ppred.append(num)
            # # TODO：注释，换一种写法
            # if query_label[i][0] == 1:
            #     # recall rate
            #     recall.append(num)
            # if num == 1:
            #     # precision rate
            #     precision.append(query_label[i][0])

        ppred = np.array(ppred)
        ppred = ppred.flatten()
        query_label = query_label.flatten()
        # print("query_label.flatten():")
        # print(query_label, '\n')

        Tp = 0
        Fp = 0
        Tn = 0
        Fn = 0
        for label, pred in zip(query_label, ppred):
            if (label == 1) and (pred == 1):
                Tp = Tp + 1
            elif (label == 1) and (pred == 0):
                Fn = Fn + 1
            elif (label == 0) and (pred == 0):
                Tn = Tn + 1
            elif (label == 0) and (pred == 1):
                Fp = Fp + 1
            else:
                print('something weird with labels')
                return -1
        # print("TP:", Tp)
        # print("FP:", Fp)
        # print("TN:", Tn)
        # print("FN:", Fn)
        # print()

        try:
            acc = (Tp + Tn) / (Tp + Tn + Fp + Fn)
        except:
            acc = 0
        try:
            precision = Tp / (Tp + Fp)
        except:
            precision = 0
        try:
            recall = Tp / (Tp + Fn)
        except:
            recall = 0
        try:
            TNR = Tn / (Fp + Tn)
        except:
            TNR = 0
        try:
            FPR = Fp / (Fp + Tn)
        except:
            FPR = 0
        try:
            FNR = Fn / (Tp + Fn)
        except:
            FNR = 0
        try:
            F1_score = 2 * precision * recall / (precision + recall)
        except:
            F1_score = 0
        # print("Accuracy:", acc)
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("TNR:", TNR)
        # print("FPR:", FPR)
        # print("FNR", FNR)
        # print("F1-score:", F1_score)
        # print()

        # print("precision:")
        # print(precision, '\n')

        # print("recall:")
        # print(recall, '\n')

        # acc = tf.cast(ppred == query_label.flatten(), tf.float32)
        # acc = tf.reduce_mean(acc)

        confusionMatrix = confusion_matrix(query_label, ppred)
        # target_names = ["Non-vulnerable","Vulnerable"]
        # classificationReport = classification_report(query_label, ppred, target_names=target_names)
        # print(confusionMatrix)
        # print(classificationReport)
        # 把权重重新设置回来
        self.meta_model.set_weights(meta_weights)
        # return loss, acc, recall, precision, confusionMatrix

        # 计算测试时间
        # 每个样本的平均测试时间
        test_query_end_time = datetime.datetime.now()
        test_query_time = test_query_end_time - test_query_start_time
        mean_test_query_time = test_query_time / logits.shape[0]

        return (
            loss,
            acc,
            recall,
            precision,
            TNR,
            FPR,
            FNR,
            F1_score,
            confusionMatrix,
            test_support_time,
            mean_test_query_time,
        )
