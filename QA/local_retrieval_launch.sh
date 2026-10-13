#!/bin/bash

echo "retriever use python:"
which python


index_file=/home/users/ntu/zihan021/scratch/PMTS-GRPO/data/wiki_dump/e5_Flat.index
corpus_file=/home/users/ntu/zihan021/scratch/PMTS-GRPO/data/wiki_dump/wiki-18.jsonl
retriever_name=e5
retriever_path=/home/users/ntu/zihan021/scratch/PMTS-GRPO/data/e5-base-v2

PYTHONUNBUFFERED=1 python search_r1/search/retrieval_server.py --index_path $index_file \
                                                               --corpus_path $corpus_file \
                                                               --topk 3 \
                                                               --retriever_name $retriever_name \
                                                               --retriever_model $retriever_path \
                                                               # --faiss_gpu
