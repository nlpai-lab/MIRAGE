"""
Evaluation script for analyzing LLM and retriever performance.

This script provides:
- Metrics: A class to calculate and report various evaluation metrics.
- Main function to run evaluations based on provided arguments.

Usage:
    python evaluation.py --do_LLM_eval <bool> --do_ret_eval <bool> --do_metrics <bool>
"""

import json, argparse, ast, os
import numpy as np
from utils import load_json, save_jsonl, str2bool
from LLM import LLM_Evaluator
from retriever import Retriever_Evaluator
from typing import List, Dict, Any, Tuple

class Metrics:
    def __init__(self, inference_dir: str, metrics_result_path: str) -> None:
        self.inference_dir = inference_dir
        self.metrics_result_path = metrics_result_path

    def get_analysis_and_report(self, base: List[Dict[str, Any]], oracle: List[Dict[str, Any]], mix: List[Dict[str, Any]], LLM_id: str, retriever_id: str, mode: str) -> Dict[str, Any]:
        result = {
            'label_count': {
                '0_0_0': 0,
                '0_0_1': 0,
                '0_1_0': 0,
                '0_1_1': 0,
                '1_0_0': 0,
                '1_0_1': 0,
                '1_1_0': 0,
                '1_1_1': 0,
            },
            'metrics': {
                'nv': 0,
                'ca': 0,
                'ci': 0,
                'cm': 0
            },
            'total': len(base)
        }

        for index in range(len(base)):
            base_label = int(base[index]['EM_label'])
            oracle_label = int(oracle[index]['EM_label'])
            mix_label = int(mix[index]['EM_label'])

            result['label_count'][f"{base_label}_{mix_label}_{oracle_label}"] += 1

            if mix_label == 0 and oracle_label == 1:
                result['metrics']['nv'] += 1
            elif mix_label == 1 and oracle_label == 1:
                result['metrics']['ca'] += 1
            elif base_label == 0 and oracle_label == 0:
                result['metrics']['ci'] += 1
            elif base_label == 1 and oracle_label == 0:
                result['metrics']['cm'] += 1

        content = {
            'LLM_name': LLM_id,
            'Retriever_name': retriever_id,
            'Given_shot': mode,
            'Scores': {
                'noise_vulnerability': round(result['metrics']['nv'] / result['total'], 5),
                'context_acceptibility': round(result['metrics']['ca'] / result['total'], 5),
                'context_insensitivity': round(result['metrics']['ci'] / result['total'], 5),
                'context_misinterpretation': round(result['metrics']['cm'] / result['total'], 5),
            },
            'Counts': result
        }
        return content

    def load_files(self, LLM_id: str, retriever_id: str, mode: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            with open(os.path.join(self.inference_dir, LLM_id + "_base_eval.json"), 'r') as f:
                base = json.load(f)
        except:
            print("Base result doesn't exist.")
            base = []

        try:
            with open(os.path.join(self.inference_dir, LLM_id + "_oracle_eval.json"), 'r') as f:
                oracle = json.load(f)
        except:
            print("Oracle result doesn't exist.")
            oracle = []

        try:
            with open(os.path.join(self.inference_dir, LLM_id + "_" + retriever_id + "_" + mode + "_eval.json"), 'r') as f:
                mix = json.load(f)
        except:
            print(f"{LLM_id}_{retriever_id}_{mode} file doesn't exist.")
            mix = []

        return base, oracle, mix

    def calculate_score(self, result_dict: Dict[str, Any]) -> float:
        return (-result_dict['Scores']['noise_vulnerability']
                + result_dict['Scores']['context_acceptibility']
                - result_dict['Scores']['context_insensitivity']
                - result_dict['Scores']['context_misinterpretation'])

    def sort_top_k(self, results: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
        results = [result for result in results if result['Given_shot'] == 'top5']

        for result_dict in results:
            result_dict['Scores']['total'] = round(self.calculate_score(result_dict), 5)

        sorted_dicts = sorted(results, key=lambda x: x['Scores']['total'], reverse=True)

        print(f"The best {k} results are as follows:")
        for i in range(k):
            temp_dict = sorted_dicts[i]
            print(f"{i + 1}. {temp_dict['LLM_name']} / {temp_dict['Retriever_name']} / {temp_dict['Given_shot']} : {temp_dict['Scores']['total']}")

        return sorted_dicts[:k]

    def evaluate(self) -> None:
        collected_results = []
        eval_files = [file for file in os.listdir(self.inference_dir) if file.endswith("_eval.json")]

        for file in eval_files:
            classes = file.split('_')
            LLM_id = classes[0]

            if len(classes) > 3:
                retriever_id = classes[1]
                mode = classes[2]

                base, oracle, mix = self.load_files(LLM_id, retriever_id, mode)
                report = self.get_analysis_and_report(base, oracle, mix, LLM_id, retriever_id, mode)
                collected_results.append(report)

        save_jsonl(collected_results, self.metrics_result_path)
        self.sort_top_k(collected_results, min(20, len(collected_results)))

def main(args: Any) -> None:
    if args.do_LLM_eval:
        LLM_evaluator = LLM_Evaluator(args.inference_dir, args.LLM_result_path)
        LLM_evaluator.evaluate()

    if args.do_ret_eval:
        RET_evaluator = Retriever_Evaluator(args.retrieval_dir, args.RET_result_path)
        RET_evaluator.evaluate()

    if args.do_metrics:
        metrics = Metrics(args.inference_dir, args.metrics_result_path)
        metrics.evaluate()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_LLM_eval', type=str2bool, default=True)
    parser.add_argument('--do_ret_eval', type=str2bool, default=True)
    parser.add_argument('--do_metrics', type=str2bool, default=True)
    parser.add_argument('--inference_dir', type=str, default="Inference_result/")
    parser.add_argument('--retrieval_dir', type=str, default="Retrieval_result/")
    parser.add_argument("--LLM_result_path", type=str, default="Evaluation_result/LLM_result.jsonl")
    parser.add_argument("--RET_result_path", type=str, default="Evaluation_result/RET_result.jsonl")
    parser.add_argument("--metrics_result_path", type=str, default="Evaluation_result/Metrics.jsonl")

    args = parser.parse_args()
    main(args)