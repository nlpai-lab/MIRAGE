"""
Classes for retrieving and evaluating document chunks.

This module provides:
- Retriever: A class to retrieve relevant document chunks.
- Retriever_Evaluator: A class to evaluate the retrieved chunks.

Usage:
    from retriever import Retriever, Retriever_Evaluator
"""

import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from utils import save_json, load_json, check_file, save_jsonl, convert_doc_pool, convert_oracle
from typing import List, Dict, Any, Tuple
import numpy as np
from datasets import load_dataset

hf_dataset = load_dataset("nlpai-lab/mirage")['train']

class Retriever:
    def __init__(self, retriever_info: Dict[str, Any]) -> None:
        self.retriever_repo = retriever_info['retriever_repo']
        self.retriever_batch_size = retriever_info['retriever_batch_size']
        self.use_cuda = retriever_info['use_cuda']
        self.top_k = retriever_info['top_k']
        self.save_directory = retriever_info['save_directory']

        self.retriever_id = self.retriever_repo.split('/')[1]
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.model = SentenceTransformer(self.retriever_repo, trust_remote_code=True).to(self.device)
        self.doc_pool = convert_doc_pool(hf_dataset)
        self.dataset = hf_dataset.to_list()
        self.file_name = f"{self.retriever_id}_top{self.top_k}.json"
        self.save_path = os.path.join(self.save_directory, self.file_name)

    def __enter__(self) -> 'Retriever':
        """Enter the context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context and release resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            print("Resources have been released.")

    def process_data(self, dataset: List[Dict[str, Any]], doc_pool: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        doc_chunks = [data_dict['doc_chunk'] for data_dict in doc_pool]
        doc_queries = [data_dict['query'] for data_dict in dataset]
        return doc_chunks, doc_queries
    
    def batch_process(self, texts: List[str]) -> torch.Tensor:
        current_batch_size = self.retriever_batch_size

        while current_batch_size > 1:
            try:
                with torch.no_grad():
                    embeddings = self.model.encode(
                        texts,
                        batch_size=current_batch_size,
                        device=self.device,
                        convert_to_tensor=True
                    )
                break
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"Out of memory error with batch size {current_batch_size}. Reducing batch size by half.")
                    current_batch_size = max(1, current_batch_size // 2)
                    torch.cuda.empty_cache()  # Clear cache to free up memory
                else:
                    raise e  # Re-raise the exception if it's not an OOM error

        return embeddings
    
    def sim_search(self, chunk_embeddings: torch.Tensor, query_embeddings: torch.Tensor) -> List[Dict[str, Any]]:
        results = []
        similarity_matrix = torch.matmul(query_embeddings, chunk_embeddings.T)
        
        for i in range(similarity_matrix.size(0)):
            # Get top-k scores and indices
            topk_scores, topk_indices = torch.topk(similarity_matrix[i], self.top_k)

            # Store results
            result = {
                'query_id': self.dataset[i]['query_id'],
                'top_chunks': [self.doc_pool[topk_index] for topk_index in topk_indices.tolist()],
                'scores': topk_scores.tolist()
            }
            results.append(result)

        return results

    def retrieve(self) -> None:
        print("***** Initiating Retrieval Process. *****")
        print("     Checking for existing result...")
        if check_file(self.save_path):
            return
        print("     Processing data for retrieval...")
        doc_chunks = [data_dict['doc_chunk'] for data_dict in self.doc_pool]
        doc_queries = [data_dict['query'] for data_dict in self.dataset]
        print("     Calculating embedding vectors...")
        chunk_embeddings = self.batch_process(doc_chunks)
        query_embeddings = self.batch_process(doc_queries)
        print(f"    Searching for top{self.top_k} results...")
        results = self.sim_search(chunk_embeddings, query_embeddings)
        print(f"    Saving the retrieval results...")
        save_json(results, self.save_path)
        print(f"    {self.file_name} has been successfully saved.")        

class Retriever_Evaluator:
    def __init__(self, retrieval_dir: str, RET_result_path: str) -> None:
        self.eval_needed = os.listdir(retrieval_dir)
        self.retrieval_dir = retrieval_dir
        self.RET_result_path = RET_result_path
        self.doc_pool = convert_doc_pool(hf_dataset)
        self.cached_gt = self.cache_ground_truth(self.doc_pool)

    def calculate_f1(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calculate_dcg(self, relevances: List[float], k: int) -> float:
        relevances = np.asfarray(relevances)[:k]
        if relevances.size:
            return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
        return 0.0

    def calculate_ndcg(self, relevances: List[float], k: int) -> float:
        dcg = self.calculate_dcg(relevances, k)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self.calculate_dcg(ideal_relevances, k)
        return dcg / idcg if idcg > 0 else 0.0

    def cache_ground_truth(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        cached_gt = {}
        for data_dict in data:
            query_id = data_dict['mapped_id']
            if query_id not in cached_gt:
                cached_gt[query_id] = []
            cached_gt[query_id].append(data_dict)
        return cached_gt

    def evaluate_results_at_k(self, query_id: str, cached_gt: Dict[str, List[Dict[str, Any]]], ret: Dict[str, Any], top_k: List[int] = [1, 3, 5]) -> Dict[int, Dict[str, float]]:
        results = {}
        relevances = [1 if query_id == chunk['mapped_id'] and chunk['support'] == 1 else 0 for chunk in ret['top_chunks']]
        gt_chunks = [chunk for chunk in cached_gt.get(query_id, []) if chunk['support'] == 1]
        
        for k in top_k:
            top_chunks = ret['top_chunks'][:k]
            relevant_chunks = [chunk for chunk in top_chunks if query_id == chunk['mapped_id'] and chunk['support'] == 1]

            precision_at_k = len(relevant_chunks) / k
            recall_at_k = len(relevant_chunks) / len(gt_chunks) if len(gt_chunks) > 0 else 0
            
            f1_at_k = self.calculate_f1(precision_at_k, recall_at_k)
            ndcg_at_k = self.calculate_ndcg(relevances, k)
            
            results[k] = {'F1': f1_at_k, 'NDCG': ndcg_at_k, 'precision': precision_at_k, 'recall': recall_at_k}
        
        return results

    def evaluate(self) -> None:
        collected_results = []
        for file in self.eval_needed:
            retriever_id = file.split('_')[0]
            rets = load_json(self.retrieval_dir + file)

            all_results = {1: [], 3: [], 5: []}
            
            for ret in rets:
                query_id = ret['query_id']
                results = self.evaluate_results_at_k(query_id, self.cached_gt, ret)
                for k, metrics in results.items():
                    all_results[k].append(metrics)
            
            avg_results = {}
            for k in [1, 3, 5]:
                avg_f1 = np.mean([result['F1'] for result in all_results[k]])
                avg_ndcg = np.mean([result['NDCG'] for result in all_results[k]])
                avg_precision = np.mean([result['precision'] for result in all_results[k]])
                avg_recall = np.mean([result['recall'] for result in all_results[k]])
                avg_results[k] = {'F1': f"{avg_f1:.5f}", 'NDCG': f"{avg_ndcg:.5f}", 'precision': f"{avg_precision:.5f}", 'recall': f"{avg_recall:.5f}"}
            
            collected_results.append({
                "Retriever_name": retriever_id,
                "Average_results": avg_results
            })
        save_jsonl(collected_results, self.RET_result_path)