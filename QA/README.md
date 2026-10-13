<h1 align="center" style="margin-top: 10px;">Tree-Based Reinforcement Learning for Credit-Precise Data Agents</h1>

## Quick Start

### Installation

#### RL Environment

```bash
conda create -n adtree python=3.12.9
conda activate adtree
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0  
pip install transformers==4.51.3 ray[default]==2.43.0 vllm==0.8.5.post1
# verl
pip install -e .
# flash attention 2
# pip install flash-attn --no-build-isolation
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install swanlab==0.7.0
```

### Retriever Environment

```bash
conda create -n retriever python=3.10.13
conda activate retriever
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0  
pip install transformers datasets pyserini
# install the gpu version faiss to guarantee efficient RL rollout
pip install faiss-gpu==1.7.3
# API function
pip install uvicorn fastapi
```

### Dataset

Download the wiki dump indexing and corpus.

```bash
save_path=/the/path/to/save
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

Process the single-hop QA and multi-hop QA datasets into parquet.

_Modify the directory paths in the scripts first._

```bash
bash scripts/data_process/data_process_multihop.sh
bash scripts/data_process/data_process_singlehop.sh
```

### RL Training

Launch a local retrieval server for single-hop QA and multi-hop QA.

_Modify the directory paths in the script first._

```bash
conda activate retriever
bash local_retrieval_launch.sh
```

Run GRPO training with Qwen2.5-3b on multi-hop QA setting.

_Modify the directory paths in the scripts first._

```bash
conda activate adtree
bash train_multihopqa_grpo.sh
```

Run DPTree training with Qwen3-4B-Base on multi-hop QA setting.

```bash
bash train_multihopqa_dptree.sh
```

## Acknowledgement

The codebase is built upon [Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO), and based on [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [veRL](https://github.com/volcengine/verl).
We express our gratitude to these open-source projects.
