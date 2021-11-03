"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
from sympy import simplify
import torch
import torch.distributed as dist
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle
import logging
import pandas as pd
import copy
import io
import sys

o_path = os.getcwd()
sys.path.append('..')

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from pytorch_pretrained_bert.modeling_mwp import BertForSeq2SeqDecoder, BertForSequenceClassification
import biunilm.seq2seq_loader_mwp as seq2seq_loader

from biunilm.eval import eval_main, eval_5fold
from utils.tool import isnumber
from process_studata import PREFIX

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    # print(text)


def split_digit(args, data_tokenizer, input_lines_primary, max_src_length):
    """将输入tokenize化，如果args.is_single_char为true，即同时拆分数字"""
    if args.is_single_char:
        input_lines = []
        for src in input_lines_primary:
            pri_src_tk = data_tokenizer.tokenize(src.lower().strip())
            pri_src_tk = [pri_src.replace('#', '') for pri_src in pri_src_tk]
            src_tk = []
            for pri_src in pri_src_tk:  # 拆分数字
                if pri_src != '[UNK]' and len(pri_src) > 1:
                    new_src = []
                    for s in pri_src:
                        new_src.append(s)
                    src_tk.extend(new_src)
                else:
                    src_tk.append(pri_src)
            input_lines.append(src_tk[:max_src_length])
    else:
        input_lines = [data_tokenizer.tokenize(
            x)[:max_src_length] for x in input_lines_primary]
    return input_lines


def merge_prefix_fraction(question, expression):
    """将question中的分式进行合并，目的是方便进行表达式长度的统计"""
    fractions, new_equation = [], []
    i = 1
    while i < len(question):
        if question[i] == '/' and i+1 < len(question) and isnumber(question[i-1]) and isnumber(question[i+1]):
            fractions.append(''.join(question[i-1:i+2]))
            i += 2
        else:
            i += 1
    j = 0
    while j < len(expression):
        if j+2 < len(expression) and expression[j+1]+expression[j]+expression[j+2] in fractions:
            new_equation.append(expression[j+1]+expression[j]+expression[j+2])
            j += 3
        else:
            new_equation.append(expression[j])
            j += 1
        
    return new_equation


def main_generation(args, logger, i_epoch, start_num=0, end_num=10000):

    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # n_gpu = 0
    
    n_gpu = torch.cuda.device_count()
    print('the count of n_gpu:', n_gpu)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)  # BertTokenizer实例化，from_pretrained作用是下载词表；add by sfeng
    vocabs = list(tokenizer.vocab.keys())

    if args.add_memory_module:
        tokenizer.max_len = args.max_analogy_len
    else:
        tokenizer.max_len = args.max_seq_length

    pair_num_relation = 0
    bi_uni_pipeline = []
    bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqBaseline(args.max_pred, list(tokenizer.vocab.keys(
            )), tokenizer.convert_tokens_to_ids, args.max_seq_length, truncate_config={'max_len_a': args.max_len_a, 'max_len_b': args.max_len_b}, 
            max_analogy_len=args.max_analogy_len, is_train=False))  

    bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seq(args.max_pred, list(tokenizer.vocab.keys(
                )), tokenizer.convert_tokens_to_ids, args.max_seq_length, truncate_config={'max_len_a': args.max_len_a, 'max_len_b': args.max_len_b}, 
                max_analogy_len=args.max_analogy_len, is_train=False, copynet_name=args.copynet_name))  

    amp_handle = None

    # Prepare model
    cls_num_labels = 2
    # type_vocab_size = 6 + \
    #     (1 if args.s2s_add_segment else 0) if args.new_segment_ids else args.type_vocab_size
    if args.add_memory_module:
        type_vocab_size = 4
        # args.max_position_embeddings = 512
    else:
        type_vocab_size = 6 + \
            (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
        # args.max_position_embeddings = 256

    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])

    def _get_token_id_set(s):
        r = None
        if s:
            w_list = []
            for w in s.split('|'):
                if w.startswith('[') and w.endswith(']'):
                    w_list.append(w.upper())
                else:
                    w_list.append(w)
            r = set(tokenizer.convert_tokens_to_ids(w_list))
        return r

    forbid_ignore_set = _get_token_id_set(args.forbid_ignore_word)  # 改成gen_and_copy_output的Index
    not_predict_set = _get_token_id_set(args.not_predict_token)
    update_model_recover_path = args.model_recover_path.replace('epoch', str(i_epoch))  # 替换具体的epoch数量；
    logger.info(update_model_recover_path)
    for model_recover_path in glob.glob(update_model_recover_path.strip()):
        # logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)
        model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model, state_dict=model_recover, num_labels=cls_num_labels, 
                num_rel=pair_num_relation, type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id, 
                search_beam_size=args.beam_size, length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id, 
                forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set, 
                not_predict_set=not_predict_set, ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode, 
                max_position_embeddings=args.max_position_embeddings, ffn_type=args.ffn_type, num_qkv=args.num_qkv, seg_emb=args.seg_emb, 
                pos_shift=args.pos_shift, add_memory_module=args.add_memory_module, topk=args.topk, max_len_a=args.max_len_a, 
                max_len_b=args.max_len_b, add_copynet=args.add_copynet, add_analogy_embedding=args.add_analogy_embedding, 
                vocabs=vocabs, args=args)
        del model_recover

        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1 and not args.is_debug:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        next_i = 0
        max_src_length = args.max_seq_length - 2 - args.max_tgt_length
        data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer

        valid_dataset = seq2seq_loader.Seq2SeqDataset(
            args.memory_valid_file, args.train_batch_size, data_tokenizer, 
            bi_uni_pipeline=bi_uni_pipeline, topk=args.topk, args=args, 
            question_sim_ids_path=args.valid_question_sim_ids_path)

        valid_data_list = valid_dataset.ex_list
        index_ids = valid_dataset.index_ids
        all_question_sim_ids = valid_dataset.question_sim_ids
        id_questions = valid_dataset.id_questions

        # 统计表达式转先序后的序列长度，以此作为题目难度的依据（序列越长难度越大）
        Prefix = PREFIX()
        if args.dataset in ['math23k', 'math23k_random', 'math23k_test1k', 'ape210k']:
        # if False: # 暂不使用
            for i in range(len(valid_data_list)):
                # expression = eval(valid_data_list[i][0][1][0])
                expression = valid_data_list[i][0][1][0]
                question = valid_data_list[i][0][0][0]
                expression = detokenize(expression)
                expression = Prefix.merge_decimal(expression)
                expression = merge_prefix_fraction(question, expression)
                valid_data_list[i][0][1][1] = len(expression)
        else:
            i = 0
            fail2prefix = 0
            while i < len(valid_data_list):
                expression = valid_data_list[i][0][1][0] 
                question = valid_data_list[i][0][0][0]
                try:
                    answer = str(simplify(expression))
                    prefix_expression = Prefix.expression2prefix(expression, answer, single_char=False)
                    if prefix_expression:
                        # valid_data_list[i][1][1] = len(Prefix.merge_decimal(prefix_expression))
                        expression = Prefix.merge_decimal(prefix_expression)
                        expression = merge_prefix_fraction(question, expression)
                        valid_data_list[i][0][1][1] = len(expression)
                    else:
                        valid_data_list[i][0][1][1] = 0
                        fail2prefix += 1
                except:
                    valid_data_list[i][0][1][1] = 0
                    fail2prefix += 1
                i += 1

        if args.easy_to_hard:
            # input_lines = sorted(list(enumerate(valid_data_list)),
            #                 key=lambda x: len(x[1][0][0]))  # 按question长度进行难度划分
            input_lines = sorted(list(enumerate(valid_data_list)),
                            key=lambda x: x[1][0][1][1])   # 按equation长度进行难度划分
        else:
            # input_lines = sorted(list(enumerate(valid_data_list)),
            #                 key=lambda x: -len(x[1][0][0]))
            input_lines = sorted(list(enumerate(valid_data_list)),
                            key=lambda x: -x[1][0][1][1])

        input_lines = input_lines[start_num:end_num]
        # input_lines = input_lines[700:end_num]  # TODO: delete
        expression_length = [line[0][1][1] for _, line in input_lines]
        easy_nums = sum([1 if length <= 3 else 0 for length in expression_length])
        medium_nums = sum([1 if length == 5 else 0 for length in expression_length])
        upper_nums = sum([1 if length == 7 else 0 for length in expression_length])
        hard_nums = sum([1 if length >= 9 else 0 for length in expression_length])
        diff_nums = [0, easy_nums, easy_nums+medium_nums, easy_nums+medium_nums+upper_nums, easy_nums+medium_nums+upper_nums+hard_nums]
        logger.info('easy_nums:{}, medium_nums:{}, upper_nums:{}, hard_nums:{}'.format(easy_nums, medium_nums, upper_nums, hard_nums))
        # assert len(expression_length) == easy_nums+medium_nums+upper_nums+hard_nums, 'difficulty numbers dont match'


        total_batch = math.ceil(len(input_lines) / args.batch_size)
        target_equations = []
        pred_equations = []
        pri_questions = []
        pred_ques_ids = []
        all_retrieve_simids = {}
        id_ques_equations = {}

        with tqdm(total=total_batch) as pbar:
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]
                next_i += args.batch_size
                buf_id = [x[0] for x in _chunk]
                ques_id = [index_ids[x[0]] for x in _chunk]
                pred_ques_ids.extend(ques_id)

                buf = [x[1][0][0][0] for x in _chunk]
                pri_questions.extend(buf)
                target_equations.extend([x[1][0][1][0] for x in _chunk])

                id_ques_equations.update({id: [ques, equ] for id, ques, equ in zip(pred_ques_ids, pri_questions, target_equations)})

                new_instances = []
                proc = bi_uni_pipeline[0]
                for idx, instances in _chunk:
                    for i, instance in enumerate(instances):
                        if i == 0:
                            instance[1] = [None, None]
                        pro_instance = proc(instance)
                        id = all_question_sim_ids[idx][i]
                        if len(pro_instance) > 1:
                            new_instances.append([id] + pro_instance)
                        else:
                            new_instances.append([id, pro_instance])

                with torch.no_grad():
                    batch = seq2seq_loader.raw_batch_list_to_batch_tensors(new_instances, is_train=False)  
                    batch = [t.to(device) if t is not None and not isinstance(t[0], str) else t for t in batch]
                    # batch, oov_tokens = seq2seq_loader.batch_list_to_batch_tensors(new_instances, is_train=False)  
                    # batch = [t.to(device) if t is not None else None for t in batch]

                    # qs_sim_ids, input_ids = batch
                    # traces, oov_tokens = model(input_ids, question_sim_ids=qs_sim_ids, 
                    #                     bi_uni_pipeline=bi_uni_pipeline[1], id_questions=id_questions)  

                    qs_sim_ids, input_ids, segment_ids, input_mask, mask_qkv, task_idx, analogy_attention_mask, position_ids, num_equ_ids, \
                        extend_oov_query_qsids, extra_zeros = batch

                    traces, oov_tokens, retrieve_simids = model(input_ids, segment_ids, position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv, 
                        analogy_attention_mask=analogy_attention_mask, num_equ_ids=num_equ_ids, 
                        extend_oov_query_qsids=extend_oov_query_qsids, extra_zeros=extra_zeros, question_sim_ids=qs_sim_ids, 
                        bi_uni_pipeline=bi_uni_pipeline[1], id_questions=id_questions)  

                    # retrieve_simids = {'question_simids':qs_simids, 'new_question_simids': new_qs_simids}
                    if all_retrieve_simids == {}:
                        all_retrieve_simids = copy.deepcopy(retrieve_simids)
                    else:
                        all_retrieve_simids['question_simids'].update(retrieve_simids['question_simids'])
                        all_retrieve_simids['new_question_simids'].update(retrieve_simids['new_question_simids'])

                    vocabs_with_oov = [vocabs+oov_token for oov_token in oov_tokens]      
                    
                    if args.beam_size > 1:

                        traces = {k: v.tolist() if not isinstance(v, list) else v for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        if args.add_copynet:
                            single_vocab = vocabs_with_oov[i]
                            output_buf = [single_vocab[w_id] for w_id in w_ids]
                        else:
                            output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                        
                        output_tokens = []
                        for t in output_buf:
                            if t in ("[SEP]", "[PAD]"):
                                break
                            output_tokens.append(t)
                        output_sequence = str(detokenize(output_tokens))
                        pred_equations.append(output_sequence)
                pbar.update(1)
                if len(pri_questions) % 200 == 0:
                    if args.Fold > 0:
                        eval_5fold(args, logger, pred_equations, target_equations, pri_questions)
                    else:
                        eval_main(args, logger, pred_equations, target_equations, pri_questions)

        if args.Fold > 0:
            eval_5fold(args, logger, pred_equations, target_equations, pri_questions)
            for i in range(len(diff_nums)-1):  # 按难度等级进行统计
                start, end = diff_nums[i], diff_nums[i+1]
                eval_5fold(args, logger, pred_equations[start:end], target_equations[start:end], pri_questions[start:end])
        else:
            eval_main(args, logger, pred_equations, target_equations, pri_questions)
            for i in range(len(diff_nums)-1):  # 按难度等级进行统计
                start, end = diff_nums[i], diff_nums[i+1]
                eval_main(args, logger, pred_equations[start:end], target_equations[start:end], pri_questions[start:end])

        # save_retrieve_result(args, all_retrieve_simids, id_questions)

    return pred_equations, target_equations, pri_questions
        

def save_retrieve_result(args, all_retrieve_simids, id_questions):
    """将经过检索器检索前后的相似题进行保存（粗排和精排结果）
    后续也可用于与随机检索之后的结果对比
    """
    Prefix = PREFIX()
    raw_question_simids = all_retrieve_simids['question_simids']
    new_question_simids = all_retrieve_simids['new_question_simids']
    result = []
    for id, new_simids in new_question_simids.items():
        raw_simids = raw_question_simids[id]
        question = ''.join(id_questions[id][0][0])
        equation = Prefix.prefix_to_postfix(Prefix.merge_decimal(id_questions[id][1][0]))
        for topk, (raw_sim_id, new_sim_id) in enumerate(zip(raw_simids, new_simids)):
            raw_sim_question = ''.join(id_questions[raw_sim_id][0][0])
            raw_sim_equation = Prefix.prefix_to_postfix(Prefix.merge_decimal(id_questions[raw_sim_id][1][0]))
            new_sim_question = ''.join(id_questions[new_sim_id][0][0])
            new_sim_equation = Prefix.prefix_to_postfix(Prefix.merge_decimal(id_questions[new_sim_id][1][0]))

            result.append([id, question, equation, topk+1, raw_sim_id, raw_sim_question, raw_sim_equation, new_sim_id, new_sim_question, new_sim_equation])

        # tmp for patent
        for topk in range(len(new_simids), len(raw_simids), 1):
            raw_sim_id = raw_simids[topk]
            raw_sim_question = ''.join(id_questions[raw_sim_id][0][0])
            raw_sim_equation = Prefix.prefix_to_postfix(Prefix.merge_decimal(id_questions[raw_sim_id][1][0]))
            new_sim_id = ''
            new_sim_question = ''
            new_sim_equation = ''
            result.append([id, question, equation, topk+1, raw_sim_id, raw_sim_question, raw_sim_equation, new_sim_id, new_sim_question, new_sim_equation])

    result = pd.DataFrame(result, columns=['id', 'question', 'equation', 'topk', 'raw_sim_id', 'raw_sim_question', 'raw_sim_equation', 'new_sim_id', 'new_sim_question', 'new_sim_equation'])
    result.to_csv(args.retrieve_result_path, index=False)

if __name__ == "__main__":
    main_generation()
