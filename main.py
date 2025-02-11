"""
Main script for running different modes of the MIRAGE project.

This script supports three modes:
- RAG: Generate and evaluate both LLM and retriever.
- LLM: Generate and evaluate only LLM.
- RET: Generate and evaluate only retriever.

Usage:
    python main.py --config <path_to_config_file>
"""

import argparse
from utils import load_yaml
from LLM import LLMGenerator
from retriever import Retriever
from typing import Any

def main(args: Any) -> None:
    config = load_yaml(args.config)
    '''
    Mode Selection
    'mode' in 'config.yaml' indicates what setup you're trying to evaluate.
    RAG : Generate and evaluate both LLM and retriever. This requires both 'LLM_info' and 'retriever_info' in 'config.yaml.'
    LLM : Generate and evaluate only LLM. This requires 'LLM_info' in config.json.
    RET : Generate and evaluate only retriever. This requires 'retriever_info' in config.yaml
    If you are not sure, we recommend running with the default code we provide.
    '''
    mode = config['mode']

    try :
        '''
        MAIN MODE: RAG Inference.
        In this step, we generate base/oracle/5-shot RAG answers.
        This requires both LLM and retriever configurations.
        Given 3 LLMs and 4 retrievers for example, this will generate 12 different RAG responses.
        During the process, we first retrieve top-5 chunks using the retriever, then iteratively generate base, oracle, and top-5 RAG responses.
        Since base and oracle responses are not retriever-dependent, they are only generated once for each LLM.
        If the mode is set for "RAG", the code will sequentially generate each response.
        '''
        if mode == "RAG":
            LLM_repos = config['LLM_info']['LLM_repo']
            retriever_repos = config['retriever_info']['retriever_repo']
            prompts = config['prompt']

            for llm_repo in LLM_repos:
                current_LLM_info = {**config['LLM_info'], 'LLM_repo': llm_repo}
                for retriever_repo in retriever_repos:
                    current_retriever_info = {**config['retriever_info'], 'retriever_repo': retriever_repo}

                    with Retriever(current_retriever_info) as retriever:
                        current_retriever_info['retriever_id'] = retriever.retriever_id
                        current_retriever_info['save_path'] = retriever.save_path
                        retriever.retrieve()
                    with LLMGenerator(current_LLM_info) as LLM:
                        LLM.generate_RAG(current_retriever_info, prompts)

        '''
        SUB-MODE-1: Retriever Search.
        In this step, we search the top-5 chunks using the retriever of your choice.
        Given each query, the retriever searches for the 5 most relevant chunks in the doc pool of 37,800 wikipedia chunks.
        This result can be used to evaluate the retriever performance, or as context for RAG configurations.
        Top-k is a adjustable parameter, however, we do not recommend increasing it too much, since the retriever pool is small.
        For RAG evaluation, Top-k should be higher or equal to 5, since the model requires up to 5 retrieved chunks to generate responses.
        '''
        if mode == "RET":
            retriever_repos = config['retriever_info']['retriever_repo']
            for retriever_repo in retriever_repos:
                current_retriever_info = {**config['retriever_info'], 'retriever_repo': retriever_repo}
                with Retriever(current_retriever_info) as retriever:
                    retriever.retrieve()

        '''
        SUB-MODE-2: LLM Inference.
        In this step, we generate base/oracle/fixed answers using the LLM of your choice.
        Then each generated answer will be labeled with EM scores for further analysis.
        This step is shared between RAG and LLM mode.
        By setting 'preset_prompt' true, you can use existing prompt to skip prompt generation.
        However, if you wish to change the prompt, modify 'prompt.yaml' and set 'preset_prompt' to false.
        '''
        if mode == "LLM":
            LLM_repos = config['LLM_info']['LLM_repo']
            prompts = config['prompt']
            for llm_repo in LLM_repos:
                current_LLM_info = {**config['LLM_info'], 'LLM_repo': llm_repo}
                with LLMGenerator(current_LLM_info) as LLM:
                    LLM.generate_LLM(prompts)
            
    
    except Exception as e:
        print(e)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.yaml')
    args = parser.parse_args()
    main(args)

