# -*- encoding: UTF-8 -*-
"""
@Author : stefan3
@Describe : pre-ranking for retrieve topN
"""

import codecs
import logging
import time
import sys, io, os
import numpy as np
import pandas as pd
import pickle
from time import time
from collections import Counter
import jieba.analyse
import jieba.posseg as pseg
from gensim import corpora, models, similarities
from gensim.models.word2vec import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append('..')

from config import arg_config
# sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')


def load_stopwords():
    stopwords = codecs.open(arg_config['path_stopword'],'r',encoding='utf8').readlines()  #对问题进行分词
    stopwords = [w.strip() for w in stopwords]
    stopwords = {sw: 1 for sw in stopwords}
    return stopwords


def load_corpus(data):
    stopwords = load_stopwords()
    original_text = [tr['original_text'] for tr in data]
    seg_text_char = [tr['seg_text_char'] for tr in data]
    # original_text = original_text[:1000]
    corpus = [tokenization(x, stopwords) for x in original_text]
    equation = [tr['equation'] for tr in data]
    ids = [tr['id'] for tr in data]
    return corpus, seg_text_char, equation, ids


def tokenization(text, stopwords):
    """
    分词
    :param text:
    :return:
    """
    # jieba.load_userdict('../file/user_dict')  #添加用户词典
    result = []
    words = pseg.cut(text)
    for word, flag in words:
        if word not in stopwords and word != ' ':   # word not in stop_word and word不为空（存在于英文question中，如mawps数据集）
            result.append(word)
    return result


def word2vec_train(corpus, dataset, fold):
    """
    word2vec模型训练
    :param corpus:
    :return:
    """
    model = Word2Vec(min_count=1)
    model.build_vocab(corpus)
    # model.train(corpus, total_examples = model.corpus_count, epochs = model.iter)
    model.train(corpus, total_examples = model.corpus_count, epochs = 100)
    if fold == 0:
        model.save(arg_config['path_{}_w2v_model'.format(dataset)])
    else:
        model.save(arg_config['path_{}_w2v_model'.format(dataset)].replace('_w2v', '_{}fold_w2v'.format(fold)))


def word2vec_sim(train_data, valid_data, topk, dataset, fold):
    """
    word2vec相似度
    """
    train_questions, train_origin_text, train_equation, train_ids = load_corpus(train_data)
    valid_questions, valid_origin_text, valid_equation, valid_ids = load_corpus(valid_data)
    train_question_sim_qsas = {}
    train_question_vectors = []
    train_sim_ids = {}
    t0 = time()
    if fold == 0:
        model = Word2Vec.load(arg_config['path_{}_w2v_model'.format(dataset)])
    else:
        model = Word2Vec.load(arg_config['path_{}_w2v_model'.format(dataset)].replace('_w2v', '_{}fold_w2v'.format(fold)))

    logging.warn('start process train questions...')
    for question in train_questions:
        question_vector = word2vec_sim_compute(question, model)  #根据单个词计算  #问题的向量
        train_question_vectors.append(question_vector)    
    
    train_batch = int(len(train_questions) / 5000) if len(train_questions) % 5000 == 0 else int(len(train_questions) / 5000) + 1
    for j in range(train_batch):
        sub_train_vec = train_question_vectors[j*5000:(j+1)*5000]
        question_sim = cosine_similarity(sub_train_vec, train_question_vectors)
        for i, qs_sim in enumerate(question_sim):
            qs_sim = qs_sim.tolist()
            self_index = qs_sim.index(max(qs_sim))
            qs_sim[self_index] = 0
            sim_qsas = []
            sim_ids = []
            for k in range(topk):  # 取出topk个相似的题目
                max_value, max_index = max(qs_sim), qs_sim.index(max(qs_sim))
                sim_qsas.append([train_origin_text[max_index], train_equation[max_index], round(max_value, 4)])
                sim_ids.append(train_ids[max_index])
                qs_sim[max_index] = 0
            primary_qsas = [train_origin_text[j*5000+i], train_equation[j*5000+i]]
            train_question_sim_qsas[train_ids[j*5000+i]] = [primary_qsas, sim_qsas]
            train_sim_ids[train_ids[j*5000+i]] = sim_ids
            if i % 100 == 0:
                logging.warn('{} question load time is:{:.2f}s'.format(j*5000+i, time()-t0))

    logging.warn('start process valid questions...')
    valid_question_sim_qsas = {}
    valid_question_vectors = []
    valid_sim_ids = {}
    for question in valid_questions:
        question_vector = word2vec_sim_compute(question, model)  #根据单个词计算  #问题的向量
        valid_question_vectors.append(question_vector)
    question_sim = cosine_similarity(valid_question_vectors, train_question_vectors)
    for i, qs_sim in enumerate(question_sim):
        qs_sim = qs_sim.tolist()
        sim_qsas = []
        sim_ids = []
        for k in range(topk):  # 取出topk个相似的题目
            max_value, max_index = max(qs_sim), qs_sim.index(max(qs_sim))
            sim_qsas.append([train_origin_text[max_index], train_equation[max_index], round(max_value, 4)])
            sim_ids.append(train_ids[max_index])
            qs_sim[max_index] = 0
        primary_qsas = [valid_origin_text[i], valid_equation[i]]
        valid_question_sim_qsas[valid_ids[i]] = [primary_qsas, sim_qsas]
        valid_sim_ids[valid_ids[i]] = sim_ids
        if i % 100 == 0:
            logging.warn('{} question load time is:{:.2f}s'.format(i, time()-t0))
            print('question:{}, equation:{}'.format(valid_origin_text[i], valid_equation[i]))
            print('sim_question:{}, sim_equation:{}, sim_value:{}'.format(sim_qsas[0][0], sim_qsas[0][1], sim_qsas[0][2]))

    return train_question_sim_qsas, valid_question_sim_qsas, train_sim_ids, valid_sim_ids


def word2vec_sim_compute(sentence, model):
    """
    word2vec相似度
    :param sentence:
    :param model:
    :return:
    """
    sentence_list = np.zeros(int(model.vector_size))
    for w in sentence:
        try:
            new_array = np.array(model[w])  #获取词向量
        except:
            new_array = np.zeros(int(model.vector_size))  #返回0向量
        sentence_list += new_array  #词向量叠加
    sentence_list_sum = sentence_list / len(sentence_list)  #句子表征向量，词向量的平均
    return sentence_list_sum


def word2vec_single_sentence_sim(question1, question2, fold, dataset):
    """计算单个句子之间的相似度"""
    if fold == 0:
        model = Word2Vec.load(arg_config['path_{}_w2v_model'.format(dataset)])
    else:
        model = Word2Vec.load(arg_config['path_{}_w2v_model'.format(dataset)].replace('_w2v', '_{}fold_w2v'.format(fold)))
    question1_vector = word2vec_sim_compute(question1, model)
    question2_vector = word2vec_sim_compute(question2, model)
    question_sim = cosine_similarity([question1_vector, question2_vector])
    return question_sim[0][1]


def word2vec_sentences_sim(questions, fold, dataset):
    """计算多个句子之间的相似度"""
    if fold == 0:
        model = Word2Vec.load(arg_config['path_{}_w2v_model'.format(dataset)])
    else:
        model = Word2Vec.load(arg_config['path_{}_w2v_model'.format(dataset)].replace('_w2v', '_{}fold_w2v'.format(fold)))
    train_question_vectors = []
    for question in questions:
        question_vector = word2vec_sim_compute(question, model)  #根据单个词计算  #问题的向量
        train_question_vectors.append(question_vector)
    question_sim = cosine_similarity(train_question_vectors)
    return question_sim


def main(data, mode='train', topk=2, dataset='math23k', fold=0):
    if mode == 'train':
        train_data = data
        train_corpus, _, _, _ = load_corpus(train_data)
        word2vec_train(train_corpus, dataset, fold)
    else:
        train_data, valid_data = data
        train_question_sim_qsas, valid_question_sim_qsas, train_sim_ids, valid_sim_ids = word2vec_sim(train_data, valid_data, topk, dataset, fold)
        return train_question_sim_qsas, valid_question_sim_qsas, train_sim_ids, valid_sim_ids
        