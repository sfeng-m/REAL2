# coding=utf-8

"""
@Company：CVTE-RESEARCH
@author: huangshifeng@cvte.com
@time: 2020/1/15
@Describe: ACC评价指标
"""

import numpy as np
import pandas as pd
import os, re, io, sys
import tqdm
import argparse
import jieba
import pickle

sys.path.append('..')
# from model_config import arg_config
from process_studata import PREFIX, is_equal

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def merge_decimal(pred):
    r_list = []
    i = 0
    while i < len(pred):
        if pred[i] == '.' and len(r_list) > 0:
            r_list[-1] = str(r_list[-1]) + str(pred[i]) + str(pred[i+1])
            i += 2
        else:
            r_list.append(pred[i])
            i += 1
    return r_list


def eval_main(args, logger, preds, tgts, pri_questions):
    Prefix = PREFIX()
    correct = 0
    question_total, acc_right, uneval = 0, 0, 0
    for pred, tgt, ques in zip(preds, tgts, pri_questions):
        # print('pred:{}, tgt:{}'.format(pred, tgt))
        question_total += 1
        pred = pred.replace('\n', '')
        tgt = tgt.replace('\n', '')
        try:
            if args.is_keep_num:
                pred_equation = eval(pred)
                pred_equation = merge_decimal(pred_equation)
            else:
                pred_equation = Prefix.split_equation(pred)
            gen_ans = Prefix.compute_prefix_expression(pred_equation)
            answer = eval(tgt.replace('[','(').replace(']',')').replace('^','**'))
            if is_equal(gen_ans, answer):
                acc_right += int(is_equal(gen_ans, answer))
                # print('...question:{}, pred_equation:{}, real_equation:{}, answer:{}'.format(''.join(ques), pred_equation, tgt, answer))
            # else:
                # logger.info('question:{}, pred_equation:{}, real_equation:{}, answer:{}'.format(''.join(ques), pred_equation, tgt, answer))
                # print('question:{}, pred_equation:{}, real_equation:{}, answer:{}'.format(''.join(ques), pred_equation, tgt, answer))
                # print('pred_answer:{}, answer:{}, uneval:{}'.format(gen_ans, answer, uneval))
        except:
            # logger.info('question:{}, pred_equation:{}, real_equation:{}, answer:{}'.format(''.join(ques), pred_equation, tgt, answer))
            # logger.info('could not eval, question:{}, pred_equation:{}, real_equation:{}, answer:{}'.format(''.join(ques), pred_equation, tgt, answer))
            # print('could not eval, question:{}, pred_equation:{}, real_equation:{}, answer:{}'.format(''.join(ques), pred_equation, tgt, answer))
            uneval += 1
    logger.info('acc_right: {}, question_total: {}, uneval: {}, correct score: {:.4f}'.format(acc_right, question_total, uneval, acc_right / question_total))


def get_convert_data():
    data_path = '../comment/datasets/mawps_data/mawps_convert_data.pkl'
    mawps_convert_data = pickle.load(open(data_path, 'rb'))
    id_digit_num = {}
    id_answer = {}
    for con_data in mawps_convert_data:
        id_digit_num[con_data['id']] = eval(con_data['ques_digit_number'])
        id_answer[con_data['id']] = con_data['ans']
    return id_digit_num, id_answer


def eval_5fold(args, logger, preds, tgts, pri_questions):
    Prefix = PREFIX()
    correct = 0
    question_total, acc_right, uneval = 0, 0, 0
    for pred, tgt, ques in zip(preds, tgts, pri_questions):
        question_total += 1
        pred = pred.replace('\n', '')
        # tgt = tgt.replace('\n', '')
        try:
            pred_equation = eval(pred)
            pred_equation = merge_decimal(pred_equation)

            gen_ans = Prefix.compute_prefix_expression(pred_equation)
            tgt_equation = merge_decimal(detokenize(tgt))
            answer = Prefix.compute_prefix_expression(tgt_equation)
            if is_equal(gen_ans, answer):
                acc_right += int(is_equal(gen_ans, answer))
        except:
            uneval += 1
    logger.info('acc_right: {}, question_total: {}, uneval: {}, correct score: {:.4f}'.format(acc_right, question_total, uneval, acc_right / question_total))


if __name__ == '__main__':
    eval_main()
