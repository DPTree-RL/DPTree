#!/bin/bash

module load miniforge3

export PYTHONUNBUFFERED=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

export TREE_SEARCH_DEBUG=true

conda activate retriever

nohup bash local_retrieval_launch.sh > retriever.log 2>&1 &

# sleep 1.5m

conda activate treegrpo

ray start --head --dashboard-host=127.0.0.1 --dashboard-port=8265 > ray.log 2>&1

bash train_singlehopqa_tree_search.sh