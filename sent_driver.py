# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 09:05:04 2021

@author: Shadow
"""

import torch 
import transformers
from datasets import load_dataset

from tokenize_data import *
from LIT_SENTIMENT import *

model_checkpoint = 'ProsusAI/finbert'
finetune_dataset = 'financial_phrasebank'

#label 2 correspnds to positive sentiment 
#label 1 is neutral 
#label 0 is negative 
train_data = load_dataset(finetune_dataset, 'sentences_75agree', split='train[:70%]')
val_data = load_dataset(finetune_dataset, 'sentences_75agree', split='train[70%:85%]')
test_data = load_dataset(finetune_dataset, 'sentences_75agree', split='train[85%:]')

#need to find the average length of the sequences
total_avg = sum( map(len, list(train_data['sentence'])) ) / len(train_data['sentence'])
print('Avg. sentence length: ', total_avg)

tokenizer = Sentiment_Tokenizer(max_length=256, tokenizer_name = model_checkpoint)

train_dataset = tokenizer.tokenize_and_encode_labels(train_data)
val_dataset = tokenizer.tokenize_and_encode_labels(val_data)
test_dataset = tokenizer.tokenize_and_encode_labels(test_data)

model = LIT_SENTIMENT(model_checkpoint = model_checkpoint,
                 hidden_dropout_prob=.1,
                 attention_probs_dropout_prob=.1,
                 save_fp='best_model.pt')

model = train_LitModel(model, train_dataset, val_dataset, max_epochs=15, batch_size=16, patience = 3, num_gpu=1)




        
    


