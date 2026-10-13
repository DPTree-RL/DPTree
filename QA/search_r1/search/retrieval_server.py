import json
import os
import warnings
from typing import List, Dict, Optional
import argparse

# 缓解显存碎片化：让 PyTorch 使用可扩展的显存段。
# 必须在 import torch / faiss 之前设置才会生效。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4,
        cache_dir="./.cache/huggingface/datasets"
    )
    return corpus

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
            results = [
                {
                    'title': content.split("\n")[0].strip("\""),
                    'text': "\n".join(content.split("\n")[1:]),
                    'contents': content
                } 
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        print("faiss reading index")
        self.index = faiss.read_index(self.index_path)
        print("faiss read index complete")
        # 提高 HNSW 搜索质量（efSearch 是运行时参数，不需重建索引）
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = 256
            print(f"Set HNSW efSearch to {self.index.hnsw.efSearch}")
        if config.faiss_gpu:
            print("faiss moving to gpus")
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            # faiss 默认每张卡预留约 1.5GB 临时显存，会把 encoder 挤爆。
            # 这里显式调小临时显存预留，给同卡上的 encoder 让出空间。
            # 配合 _index_search 的小批搜索，避免 TemporaryMemoryOverflow。
            temp_mem_mb = getattr(config, 'faiss_temp_memory_mb', 512)
            try:
                ngpu = faiss.get_num_gpus()
                self._gpu_resources = []
                for _ in range(ngpu):
                    res = faiss.StandardGpuResources()
                    if temp_mem_mb is not None and temp_mem_mb >= 0:
                        res.setTempMemory(int(temp_mem_mb) * 1024 * 1024)
                    self._gpu_resources.append(res)
                self.index = faiss.index_cpu_to_gpu_multiple_py(
                    self._gpu_resources, self.index, co
                )
                print(f"faiss move to gpus complete (temp_memory={temp_mem_mb}MB/gpu)")
            except Exception as e:
                # 回退到原始路径，至少保证能跑起来。
                warnings.warn(
                    f"custom faiss gpu resources failed ({e}), "
                    f"falling back to index_cpu_to_all_gpus"
                )
                self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
                print("faiss move to gpus complete (default temp memory)")

        print("loading corpus")
        self.corpus = load_corpus(self.corpus_path)
        print("load corpus complete")
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        # 每次调用 faiss.index.search 时最多喂入的 query 向量数量。
        # faiss GPU 的临时显存池有限，一次搜索太多向量会触发
        # TemporaryMemoryOverflow / cudaMalloc OOM，因此在这里强行切小。
        self.faiss_search_batch_size = getattr(config, 'faiss_search_batch_size', 128)

    def _index_search(self, batch_emb, num):
        """对 faiss 索引做搜索，但把 query 向量切成足够小的子批，避免 GPU 临时显存溢出。

        如果某个子批仍然 OOM，则把子批大小减半后重试（自适应下探），
        直到单条向量都放不下才真正抛错。
        """
        n = batch_emb.shape[0]
        sub_bs = min(self.faiss_search_batch_size, n) if n > 0 else 1
        sub_bs = max(1, sub_bs)
        all_scores = []
        all_idxs = []
        start = 0
        while start < n:
            end = min(start + sub_bs, n)
            chunk = batch_emb[start:end]
            try:
                s, i = self.index.search(chunk, k=num)
            except RuntimeError as e:
                msg = str(e).lower()
                is_oom = ('out of memory' in msg
                          or 'memoryoverflow' in msg
                          or 'alloc fail' in msg)
                if is_oom and chunk.shape[0] > 1:
                    # 子批仍然太大：减半后不推进 start，重试这一段。
                    sub_bs = max(1, chunk.shape[0] // 2)
                    torch.cuda.empty_cache()
                    warnings.warn(
                        f"faiss search OOM, reducing faiss sub-batch size to {sub_bs} and retrying"
                    )
                    continue
                raise
            all_scores.append(s)
            all_idxs.append(i)
            start = end
        return np.concatenate(all_scores, axis=0), np.concatenate(all_idxs, axis=0)

    def _encode(self, query_batch):
        """编码 query 向量，OOM 时自动把 batch 减半重试，避免 encoder forward 撑爆显存。"""
        try:
            return self.encoder.encode(query_batch)
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower():
                raise
            # 已经是单条却仍然 OOM，无法再切分。
            if isinstance(query_batch, str) or len(query_batch) <= 1:
                raise
            torch.cuda.empty_cache()
            mid = len(query_batch) // 2
            warnings.warn(
                f"encoder OOM, splitting encode batch of {len(query_batch)} "
                f"into {mid}+{len(query_batch) - mid} and retrying"
            )
            left = self._encode(query_batch[:mid])
            right = self._encode(query_batch[mid:])
            return np.concatenate([left, right], axis=0)

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self._encode(query)
        scores, idxs = self._index_search(query_emb, num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self._encode(query_batch)
            batch_scores, batch_idxs = self._index_search(batch_emb, num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results

def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        faiss_search_batch_size: int = 128,
        faiss_temp_memory_mb: int = 512
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.faiss_search_batch_size = faiss_search_batch_size
        self.faiss_temp_memory_mb = faiss_temp_memory_mb


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=request.return_scores
    )
    
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for computation')
    parser.add_argument('--faiss_search_batch_size', type=int, default=128,
                        help='一次喂给 faiss GPU 搜索的最大 query 向量数量，越小越省显存。')
    parser.add_argument('--faiss_temp_memory_mb', type=int, default=512,
                        help='每张 GPU 上 faiss 预留的临时显存(MB)，调小可给同卡的 encoder 让出显存。')
    parser.add_argument('--retrieval_batch_size', type=int, default=128,
                        help='encoder 一次编码的最大 query 数量，越小越省显存。')

    args = parser.parse_args()
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=args.retrieval_batch_size,
        faiss_search_batch_size=args.faiss_search_batch_size,
        faiss_temp_memory_mb=args.faiss_temp_memory_mb,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
