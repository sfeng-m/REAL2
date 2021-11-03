from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
o_path = os.getcwd()
sys.path.append('..')

from utils.Logger import initlog
from biunilm.decoder_seq2seq_mwp import main_generation
from biunilm.parser_args_batch import get_args
from biunilm.run_seq2seq_mwp import main as train_main

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
Epoch = 50

args = get_args()

def main():
    args.log_dir = './comment/math23k/model_log'
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.dataset == 'math23k':
        # for fold in [1,2,3,4,5]:
        for fold in [1]:
            args.Fold = fold
            OUTPUT_DIR = './comment/math23k/'
            args.memory_train_file = './preprocess/sim_result/sim_question_by_w2v_train_math23k_equNorm_{}fold_top10.pkl'.format(args.Fold)
            args.memory_valid_file = './preprocess/sim_result/sim_question_by_w2v_valid_math23k_equNorm_{}fold_top10.pkl'.format(args.Fold)
            args.train_question_sim_ids_path = './preprocess/raw_sim_dict/math23k_train_sim_id_{}fold.pkl'.format(args.Fold)
            args.valid_question_sim_ids_path = './preprocess/raw_sim_dict/math23k_valid_sim_id_{}fold.pkl'.format(args.Fold)
            args.ids_questions_path = './preprocess/raw_sim_dict/raw_math23k_data_dict.pkl'
            
            model_name = 'REAL2_math23k_{}fold_topn{}_top{}_{}'.format(args.Fold, args.retrieve_topn, args.topk, args.retrieve_model_name)
            log_path = model_name + '.log'
            args.output_dir = OUTPUT_DIR + model_name
            args.model_recover_path = args.output_dir + '/model.epoch.bin'
            args.retrieve_result_path = OUTPUT_DIR + 'retrieve_result/math23k_{}fold_{}.csv'.format(args.Fold, args.retrieve_model_name)
                    
            logger = initlog(logfile=args.log_dir + "/" + log_path)
            logger.info('pid:{}, epoch:{}'.format(os.getpid(), Epoch))
            logger.info('args:{}'.format(args))
            if args.is_train:
                train_main(args, logger)
            else:
                _, _, _ = main_generation(args, logger, i_epoch=Epoch)

    elif args.dataset == 'math23k_test1k':
        OUTPUT_DIR = './comment/math23k/'
        args.memory_train_file = './preprocess/sim_result/sim_question_by_w2v_train_math23k_test1k_equNorm_top10.pkl'
        args.memory_valid_file = './preprocess/sim_result/sim_question_by_w2v_valid_math23k_test1k_equNorm_top10.pkl'
        args.train_question_sim_ids_path = './preprocess/raw_sim_dict/math23k_test1k_train_sim_id_top10.pkl'
        args.valid_question_sim_ids_path = './preprocess/raw_sim_dict/math23k_test1k_valid_sim_id_top10.pkl'
        args.ids_questions_path = './preprocess/raw_sim_dict/raw_math23k_test1k_data_dict.pkl'
        
        model_name = 'REAL2_math23k_test1k_topn{}_top{}_{}'.format(args.retrieve_topn, args.topk, args.retrieve_model_name) 
        log_path = model_name + f'{args.repeat}.log'
        args.output_dir = OUTPUT_DIR + model_name
        args.model_recover_path = args.output_dir + f'/model{args.repeat}.epoch.bin'  # 重复三次取平均值

        logger = initlog(logfile=args.log_dir + "/" + log_path)
        logger.info('pid:{}, epoch:{}'.format(os.getpid(), Epoch))
        logger.info('args:{}'.format(args))
        if args.is_train:
            train_main(args, logger)
        else:
            _, _, _ = main_generation(args, logger, i_epoch=Epoch)


if __name__ == "__main__":
    main()


    

    