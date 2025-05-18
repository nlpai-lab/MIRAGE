"""
Classes for generating and evaluating responses using LLMs.

This module provides:
- LLMGenerator: A class to generate responses using various LLMs.
- LLM_Evaluator: A class to evaluate the generated responses.

Usage:
    from LLM import LLMGenerator, LLM_Evaluator
"""

import torch.distributed
import os, torch, logging, ast
from utils import load_json, save_json, save_jsonl, check_file, convert_doc_pool, convert_oracle
from vllm import LLM, SamplingParams
import contextlib
import gc
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel)
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Union
from openai import AsyncOpenAI
import asyncio
import numpy as np
from tqdm.asyncio import tqdm as async_tqdm
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
hf_dataset = load_dataset("nlpai-lab/mirage")['train']

class LLMGenerator:
    def __init__(self, LLM_info: Dict[str, Any]) -> None:
        self.LLM_repo = LLM_info['LLM_repo']
        self.vllm_configs = LLM_info['vllm_configs']
        self.preset_prompt = LLM_info['preset_prompt']
        self.save_directory = LLM_info['save_directory']

        try:
            self.LLM_id = self.LLM_repo.split('/')[1]
            self.model = LLM(model=self.LLM_repo,
                             tensor_parallel_size=self.vllm_configs['tensor_parallel_size'],
                             gpu_memory_utilization=self.vllm_configs['gpu_memory_utilization'])
        except Exception as e:
            logging.error(f"Error occurred while loading the model. {e}")
            logging.info("If the given model is set to call GPT models, ignore this message.")
            logging.info("Otherewise, please check the LLM model repository and configurations.")
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.LLM_id = self.LLM_repo
            self.model = None
            if os.getenv("OPENAI_API_KEY") is None:
                raise ValueError("OpenAI API key is not set. Please set the API key in the environment variable.")
            else:
                self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

        self.doc_pool = convert_doc_pool(hf_dataset)
        self.dataset = hf_dataset.to_list()
        self.oracle = convert_oracle(hf_dataset)

    def __enter__(self) -> 'LLMGenerator':
        """Enter the context."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context and release resources."""
        if hasattr(self, 'model') and self.model is not None:
            destroy_model_parallel()
            destroy_distributed_environment()
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
            del self.model
            torch.cuda.empty_cache()
            gc.collect()
            logging.info("Resources have been released.")
    
    async def process_prompt(self, client, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=self.LLM_id,
                    messages=prompt,
                    temperature=self.vllm_configs['sampling_params']['temperature'],
                    top_p=self.vllm_configs['sampling_params']['top_p'],
                    max_tokens=self.vllm_configs['sampling_params']['max_tokens']
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed after {max_retries} attempts: {e}")
                    return None
                await asyncio.sleep(2 ** attempt)

    async def process_prompts(self, prompts, max_concurrent=5):
        client = AsyncOpenAI(api_key=self.OPENAI_API_KEY)
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}

        async def process_with_semaphore(idx, prompt):
            async with semaphore:
                result = await self.process_prompt(client, prompt)
                results[idx] = result

        tasks = [process_with_semaphore(idx, prompt) for idx, prompt in enumerate(prompts)]
        with async_tqdm(total=len(tasks), desc="Processing prompts...") as pbar:
            for f in asyncio.as_completed(tasks):
                await f
                pbar.update(1)
        return results

    def generate_LLM_prompt(self, prompts: Dict[str, str]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        if self.preset_prompt:
            base_prompts = load_json("Preset_prompt/base.json")
            oracle_prompts = load_json("Preset_prompt/oracle.json")
            fixed_prompts = load_json("Preset_prompt/fixed.json")
            return base_prompts, oracle_prompts, fixed_prompts

        else:
            system_prompt = prompts['system_prompt']
            base_format = prompts['base_format']
            RAG_format = prompts['RAG_format']

            base_prompts = []
            oracle_prompts = []
            fixed_prompts = []

            for idx, data_dict in tqdm(enumerate(self.dataset), desc="Generating LLM prompts...", total=len(self.dataset)):
                base_prompts.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": base_format.format(query=data_dict['query'])}
                ])

                oracle_prompts.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": RAG_format.format(query=data_dict['query'], context=self.oracle[data_dict['query_id']]['doc_chunk'])}
                ])

                doc_chunks = [self.doc_pool[5*i:5*i+5] for i in range(len(self.doc_pool)//5)]
                context = ""
                for i, chunk in enumerate(doc_chunks[idx]):
                    context += f"{i+1}. {chunk['doc_chunk']}"
                fixed_prompts.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": RAG_format.format(query=data_dict['query'], context=context)}
                ])

            return base_prompts, oracle_prompts, fixed_prompts
    
    def generate_RAG_prompt(self, prompts: Dict[str, str], retrieved_chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        system_prompt = prompts['system_prompt']
        base_format = prompts['base_format']
        RAG_format = prompts['RAG_format']
        
        base_prompts = []
        oracle_prompts = []
        top_five_prompts = []

        for idx, data_dict in tqdm(enumerate(self.dataset), desc="Generating RAG prompts...", total=len(self.dataset)):
            base_prompts.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": base_format.format(query=data_dict['query'])}
                ])
            oracle_prompts.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": RAG_format.format(query=data_dict['query'], context=self.oracle[data_dict['query_id']]['doc_chunk'])}
            ])
            
            top_five_chunks = retrieved_chunks[idx]['top_chunks'][:5]
            context = ""
            for i, chunk in enumerate(top_five_chunks):
                context += f"{i+1}. {chunk['doc_chunk']}"
            top_five_prompts.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": RAG_format.format(query=data_dict['query'], context=context)}
            ])

        return base_prompts, oracle_prompts, top_five_prompts

    def generate_response(self, dataset: List[Dict[str, Any]], prompts: List[Dict[str, str]], context_setting: str) -> None:
        file_name = f"{self.LLM_id}_{context_setting}.json"
        save_path = os.path.join(self.save_directory, file_name)

        if not check_file(save_path):
            if self.model is None:  # Use OpenAI API
                responses = asyncio.run(self.process_prompts(prompts))
                for idx, result in responses.items():
                    dataset[idx]['model_predictions'] = [result]
            else:
                sampling_params = SamplingParams(**self.vllm_configs['sampling_params'])
                logging.info(sampling_params)
                responses = self.model.chat(prompts, sampling_params=sampling_params)
                for idx, response in enumerate(tqdm(responses, desc=f"Storing model responses at {save_path}...")):
                    dataset[idx]['model_predictions'] = response.outputs[0].text.strip()
            save_json(dataset, save_path)

    def generate_LLM(self, prompts: Dict[str, str]) -> None:
        logging.info("***** Initiating LLM Inference. *****")
        logging.info("     Generating prompts...")
        base_prompts, oracle_prompts, fixed_prompts = self.generate_LLM_prompt(prompts)
        logging.info("     Generating model responses...")
        self.generate_response(self.dataset, base_prompts, 'base')
        self.generate_response(self.dataset, oracle_prompts, 'oracle')
        self.generate_response(self.dataset, fixed_prompts, 'fixed')
    
    def generate_RAG(self, retriever_info: Dict[str, Any], prompts: Dict[str, str]) -> None:
        retrieved_chunks = load_json(retriever_info['save_path'])
        logging.info("***** Initiating RAG Inference. *****")
        logging.info("     Generating prompts...")
        base_prompts, oracle_prompts, top_five_prompts = self.generate_RAG_prompt(prompts, retrieved_chunks)
        logging.info("     Generating model responses...")
        self.generate_response(self.dataset, base_prompts, 'base')
        self.generate_response(self.dataset, oracle_prompts, 'oracle')
        self.generate_response(self.dataset, top_five_prompts, f'{retriever_info["retriever_id"]}_top5')

class LLM_Evaluator:
    def __init__(self, inference_dir: str, LLM_result_path: str) -> None:
        self.eval_needed = self.check_eval_needed(inference_dir)
        self.inference_dir = inference_dir
        self.LLM_result_path = LLM_result_path

    def check_eval_needed(self, dir_path: str) -> List[str]:
        files = os.listdir(dir_path)
        eval_needed = []
        for file in files:
            if file.split('.json')[0]+'_eval.json' not in files and not file.endswith('_eval.json'):
                    eval_needed.append(file)
        print("Evaluation needed for the following files: ", eval_needed)
        return eval_needed

    def check_inference_mode(self, file_name: str) -> Tuple[str, Union[str, None], str]:
        num_separator = len(file_name.split('_'))
        if num_separator == 2:
            LLM_id = file_name.split('_')[0]
            mode = file_name.split('_')[1].split('.')[0]
            return LLM_id, None, mode
        elif num_separator == 3:
            LLM_id = file_name.split('_')[0]
            retriever_id = file_name.split('_')[1]
            top_k = file_name.split('_')[2].split('.')[0]
            return LLM_id, retriever_id, top_k
        else:
            raise ValueError(f"Invalid file name format for {file_name}. The file name should be in the format of 'LLM-id_mode.json' or 'LLM-id_retriever-id_top-k.json'.")

    def calculate_EM_loose(self, answer_array: List[str], model_predictions: List[str]) -> float:
        answer_set = set(ans.lower() for ans in answer_array)
        model_prediction = model_predictions[0].lower()

        em_loose_score = 1.0 if any(ans in model_prediction for ans in answer_set) else 0.0

        return em_loose_score

    def calculate_EM_strict(self, answer_array: List[str], model_predictions: List[str]) -> float:
        answer_set = set(ans.lower() for ans in answer_array)
        model_prediction = model_predictions[0].lower()

        em_strict_score = 1.0 if any(ans == model_prediction for ans in answer_set) else 0.0

        return em_strict_score

    def calculate_f1(self, answer_array: List[str], model_predictions: List[str]) -> float:
        model_prediction = model_predictions[0].lower()

        predicted_tokens = set(model_prediction.split())
        answer_tokens = set(token for ans in answer_array for token in ans.lower().split())
        
        true_positives = len(predicted_tokens.intersection(answer_tokens))
        precision = true_positives / len(predicted_tokens) if predicted_tokens else 0
        recall = true_positives / len(answer_tokens) if answer_tokens else 0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1_score
    
    def evaluate(self) -> None:
        collected_results = []
        for eval_file in self.eval_needed:
            try:
                LLM_id, retriever_id, mode = self.check_inference_mode(eval_file)
                responses = load_json(self.inference_dir + eval_file)
                EM_loose_scores = []
                EM_strict_scores = []
                F1_scores = []

                for data_dict in responses:
                    answer_array = data_dict['answer']
                    if isinstance(answer_array, str):
                        answer_array = ast.literal_eval(answer_array)
                    model_predictions = data_dict['model_predictions']
                    if isinstance(model_predictions, str):
                        model_predictions = [model_predictions]
                        
                    EM_loose_score = self.calculate_EM_loose(answer_array, model_predictions)
                    EM_strict_score = self.calculate_EM_strict(answer_array, model_predictions)
                    data_dict['EM_label'] = EM_loose_score
                    data_dict['EM_strict'] = EM_strict_score

                    EM_loose_scores.append(EM_loose_score)
                    EM_strict_scores.append(EM_strict_score)
                    F1_scores.append(self.calculate_f1(answer_array, model_predictions))

                avg_results = {
                    "LLM_name": LLM_id,
                    "Retriever_name": retriever_id,
                    "Given_shot": mode,
                    "F1_score": round(np.mean(F1_scores), 5),
                    "EM_loose": round(np.mean(EM_loose_scores), 5),
                    "EM_strict": round(np.mean(EM_strict_scores), 5)
                }
                collected_results.append(avg_results)
                eval_file_path = os.path.join(self.inference_dir, eval_file[:-5] + "_eval.json")
                save_json(responses, eval_file_path)

            except Exception as e:
                print(f"Error occurred while evaluating {eval_file}. {e}")
                print("Skip this file and continue to the next file.")
        
        save_jsonl(collected_results, self.LLM_result_path)