# REAL2

Code for NIPS2021 Paper on MATHAI4ED Workshop: ["REAL2: An End-to-end Memory-augmented Solver for Math Word Problems"](https://mathai4ed.github.io/papers/papers/paper_7.pdf). 

REAL2: improve the effectiveness of [REAL model](https://aclanthology.org/2021.findings-emnlp.68.pdf) to solve math work problems(MWP) by optimizing the memory module.

# environment  
python3.6, pytorch1.2\
You can install related packages directly through "pip install requirements.txt" 

# preprocess data
    python3 memory_module.py

# train: 
     python3 run.py --is_train --num_train_epochs 50 \
        --start_lr_decay_epoch 25 --dataset math23k \
        --retrieve_model_name cnn --retrieve_topn 10 --topk 3 

# test:
    python3 run.py --dataset math23k  \
        --retrieve_model_name cnn --retrieve_topn 10 --topk 3 

# framework
 
<img width="800" height="480" src="https://github.com/sfeng-m/REAL2/blob/master/images/framework.png" />

# result
To investigate the effectiveness of the trainable memory module, we implemented our framework follows the settings of REAL and only modified the framework of stage 1 part.  In particular, we compare different backbone of memory module that involves TextCNN, TextRCNN, Transformer, and BERT model. 

<img width="500" height="300" src="https://github.com/sfeng-m/REAL2/blob/master/images/result.png" />


# Acknowledgments
Our code is based on [unilm](https://github.com/microsoft/unilm/tree/master/unilm-v1) . We thank the authors for their wonderful open-source efforts. We use the same license as unilm.
