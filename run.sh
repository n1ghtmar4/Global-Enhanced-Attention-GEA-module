#!/bin/bash
set -ex

	CUDA_VISIBLE_DEVICES=1,0 python run_chid+att+pre+new1.py \
	--vocab_file ./chinese_wwm_pytorch/vocab.txt \
	--bert_config_file ./chinese_wwm_pytorch/bert_config.json \
	--init_checkpoint ./chinese_wwm_pytorch/pytorch_model.bin \
	--do_predict \
	--train_file ./data/train.txt \
	--train_ans_file ./data/train_answer.csv \
	--predict_file ./data/dev.txt \
	--train_batch_size 64 \
	--predict_batch_size 128 \
	--learning_rate 2e-5 \
	--num_train_epochs 10.0 \
	--max_seq_length 256 \
	--fp16 \
	--output_dir ./output_model
