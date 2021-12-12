#!/bin/bash
export PYTHONUNBUFFERED=1
pretrain_path='bert-base-multilingual-uncased'
train_file='benchmark/disrex_dataset/disrex_train.txt'
val_file='benchmark/disrex_dataset/disrex_val.txt'
test_file='benchmark/disrex_dataset/disrex_test.txt'
rel2id_file='benchmark/disrex_dataset/rel2id.txt'
batch_size=32
max_length=512
max_epoch=5
seed=772
name=$pretrain_path'_'$max_length'_'$batch_size'_'$max_epoch'_'$seed'_word_level_disrex_replicate'
echo $name
python main.py --pretrain_path $pretrain_path --train_file $train_file --val_file $val_file --test_file $test_file --rel2id_file $rel2id_file --batch_size $batch_size --max_epoch $max_epoch --max_length $max_length --ckpt $name
