# -*- encoding: UTF-8 -*-
"""
@author: 'stefan3'
@describe: pre-ranking for retrieve topN
"""

import pickle
from process_studata import fetch_math23k
from config import arg_config
from preprocess import train_w2v
#改变标准输出的默认编码
import io, sys
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

MODE = 'train'
TopK = 10
Used_Transfer_Num = False
Used_Equ_Norm = True


def sim_model(train_data, valid_data, train_simid_path, valid_simid_path, train_simqs_path, valid_simqs_path, raw_data_dict, fold):
    """相似度模型的训练和预测"""
    if MODE == 'train':
        train_w2v.main(data=train_data, mode=MODE, topk=TopK, dataset=Dataset, fold=fold)

    train_question_sim_qsas, valid_question_sim_qsas, train_sim_ids, valid_sim_ids = train_w2v.main(
        data=[train_data, valid_data], mode='valid', topk=TopK, dataset=Dataset, fold=fold)

    pickle.dump(train_question_sim_qsas, open(train_simqs_path, 'wb'))
    pickle.dump(valid_question_sim_qsas, open(valid_simqs_path, 'wb'))
    pickle.dump(train_sim_ids, open(train_simid_path, 'wb'))
    pickle.dump(valid_sim_ids, open(valid_simid_path, 'wb'))

    def print_sim_data(input_sim_ids):
        for i, (qid, sim_ids) in enumerate(input_sim_ids.items()):
            print('qid:{}, value:{}'.format(qid, raw_data_dict[qid]))
            for sim_id in sim_ids[:5]:
                print('key:{}, value:{}'.format(sim_id, raw_data_dict[sim_id]))
            print('...'*30)
            if i == 10:
                break
    print_sim_data(train_sim_ids)
    print('****'*30)
    # print_sim_data(valid_sim_ids)


def process_fivefold_data():
    """处理五折交叉验证数据"""
    Fetch = fetch_math23k()
    all_data, equ_ids_dict, raw_data_dict = Fetch.load_math23k_line_data(filename=arg_config['path_math23k_dataset'], mode='train', single_char=False, used_equ_norm=Used_Equ_Norm)
    pickle.dump(raw_data_dict, open(arg_config['raw_math23k_data_dict_path'], 'wb'))
    
    one_fold_length = int(len(all_data) / 5)
    for i in range(5):
        fold_valid_data = all_data[one_fold_length*i:one_fold_length*(i+1)]
        fold_train_data = all_data[:one_fold_length*i] + all_data[one_fold_length*(i+1):]

        train_simid_path = arg_config['math23k_train_sim_id_path'].replace('.pkl', '_{}fold.pkl'.format(i+1))
        valid_simid_path = arg_config['math23k_valid_sim_id_path'].replace('.pkl', '_{}fold.pkl'.format(i+1))
        train_simqs_path = arg_config['path_w2v_sim_question_train'].replace('.pkl', '_{}_equNorm_{}fold_top{}.pkl'.format(Dataset, i+1, TopK))
        valid_simqs_path = arg_config['path_w2v_sim_question_valid'].replace('.pkl', '_{}_equNorm_{}fold_top{}.pkl'.format(Dataset, i+1, TopK))
        
        sim_model(fold_train_data, fold_valid_data, train_simid_path, valid_simid_path, train_simqs_path, valid_simqs_path, raw_data_dict, fold=i+1)
        

def process_unfold_data():
    """处理无需进行交叉验证的数据集，主要处理math23k test1k官方测试集"""

    Fetch = fetch_math23k()
    train_data, equ_ids_dict, raw_data_dict = Fetch.load_math23k_line_data(filename=arg_config['path_math23k_all_train'], mode='train', single_char=False, used_equ_norm=Used_Equ_Norm)
    valid_data, _, _ = Fetch.load_math23k_line_data(filename=arg_config['path_math23k_test_1000'], mode='train', single_char=False, used_equ_norm=Used_Equ_Norm)
    pickle.dump(raw_data_dict, open(arg_config['raw_math23k_test1k_data_dict_path'], 'wb'))
    train_simid_path = arg_config['math23k_test1k_train_sim_id_path'].replace('.pkl', '_top{}.pkl'.format(TopK))
    valid_simid_path = arg_config['math23k_test1k_valid_sim_id_path'].replace('.pkl', '_top{}.pkl'.format(TopK))
    train_simqs_path = arg_config['path_w2v_sim_question_train'].replace('.pkl', '_{}_equNorm_top{}.pkl'.format(Dataset, TopK))
    valid_simqs_path = arg_config['path_w2v_sim_question_valid'].replace('.pkl', '_{}_equNorm_top{}.pkl'.format(Dataset, TopK))
    
    sim_model(train_data, valid_data, train_simid_path, valid_simid_path, train_simqs_path, valid_simqs_path, raw_data_dict, fold=0)
    

if __name__ == '__main__':
    # Dataset = 'math23k'
    # process_fivefold_data()
    Dataset = 'math23k_test1k'    
    process_unfold_data()
    

    

    
