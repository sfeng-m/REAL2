import pickle


def gain_stage1_perfName():
    """将数据转换成模型输入的格式"""
    df = pd.read_csv(arg_config['path_fit_perfName_label'])
    df.columns = ['id', 'question', 'answer']
    f_train_src = open(arg_config['path_perfName_classify_train_src'], 'w', encoding='utf-8')
    f_train_tgt = open(arg_config['path_perfName_classify_train_tgt'], 'w', encoding='utf-8')
    f_valid_src = open(arg_config['path_perfName_classify_valid_src'], 'w', encoding='utf-8')
    f_valid_tgt = open(arg_config['path_perfName_classify_valid_tgt'], 'w', encoding='utf-8')
    train_length = int(len(df)*0.9)
    question_answers = df[['question', 'answer']].values
    random.shuffle(question_answers)
    for i, (question, answer) in enumerate(question_answers):
        # perfName, perfLabel = row[0], row[1]
        if i <= train_length:
            f_train_src.write(question.replace('\n', '')+'\n')
            f_train_tgt.write(answer.replace('\n', '')+'\n')
        else:
            f_valid_src.write(question.replace('\n', '')+'\n')
            f_valid_tgt.write(answer.replace('\n', '')+'\n')
    f_train_src.close()
    f_train_tgt.close()
    f_valid_src.close()
    f_valid_tgt.close()

    test_df = pd.read_csv(arg_config['path_fit_perfName_label'])
    test_df.columns = ['id', 'question', 'answer']
    f_test_src = open(arg_config['path_perfName_classify_valid_src'], 'w', encoding='utf-8')
    f_test_tgt = open(arg_config['path_perfName_classify_valid_tgt'], 'w', encoding='utf-8')
    question_answers = test_df[['question', 'answer']].values
    for i, (question, answer) in enumerate(question_answers):
        f_test_src.write(question.replace('\n', '')+'\n')
        f_test_tgt.write(answer.replace('\n', '')+'\n')
    f_test_src.close()
    f_test_tgt.close()


def judge_train_test_data():
    tmp_data_train = pickle.load(open('./data_debug_train.pkl', 'rb'))
    tmp_data_test = pickle.load(open('./data_debug_test.pkl', 'rb'))
    for key, value in tmp_data_train.items():
        if key == 'args':
            if value != tmp_data_train['args']:
                print('key:{}'.format(key))
                print('train_value:{}'.format(value))
                print('test_value:{}'.format(tmp_data_test[key]))
        else:
            if value.size(1) != tmp_data_test[key].size(1):
                same_len = (tmp_data_train[key][:,:448]==tmp_data_test[key][:,:448]).sum()
            else:
                same_len = (tmp_data_train[key]==tmp_data_test[key]).sum()
            all_len = tmp_data_test[key].size(0) * tmp_data_test[key].size(1)
            if same_len < all_len:
                print('key:{}, same_len:{}, all_len:{}'.format(key, same_len, all_len))
                print('train_value:{}'.format(value))
                print('test_value:{}'.format(tmp_data_test[key]))


if __name__ == '__main__':
    judge_train_test_data()