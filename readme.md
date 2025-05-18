# MIRAGE Benchmark

 [![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/nlpai-lab/mirage) [![Read the Paper](https://img.shields.io/badge/Paper-arXiv%3A2054.17137-B31B1B.svg)](https://arxiv.org/abs/2504.17137)

:tada: Our paper introducing MIRAGE has been accepted to the Findings of NAACL 2025!

MIRAGE is a benchmark designed to evaluate the performance of retrieval-augmented generation (RAG) systems using various QA datasets. It includes 7560 Q/A pairs and 37800 context pools collected from Wikipedia-based QA benchmarks such as IfQA, NaturalQA, TriviaQA, DROP, and PopQA.


## Key Features

- **RAG Evaluation**: Measures the robustness of LLMs in RAG environments using three setups:
  - **Base**: Closed-book QA where only the query is provided.
  - **Oracle**: Open-book QA with the correct context provided.
  - **Mixed**: Realistic RAG environment with both correct and noisy contexts.
- **MIRAGE Metrics**: Evaluates LLM adaptability in RAG environments through:
  - **Noise Vulnerability**: Assesses the model's susceptibility to noise in the context.
  - **Context Acceptability**: Evaluates the model's ability to effectively leverage the provided context to generate accurate answers.
  - **Context Insensitivity**: Highlights cases where the model fails to utilize the context information.
  - **Context Misinterpretation**: Identifies cases where the model answers correctly without context but hallucinates when given the oracle context.

## Evaluation Insights

- **Retriever Dependency**: Noise Vulnerability and Context Acceptability metrics show significant differences based on the retriever used, indicating that the retrieval phase is a bottleneck in RAG pipelines.
- **LLM Capability**: Context Insensitivity and Context Misinterpretation metrics are more related to the inherent capabilities of the LLM, showing improvements with newer models.

## Retriever Evaluation

- **Efficient Evaluation**: Uses a retrieval pool of 37.8k chunks (1% of the full wiki-dump) to significantly reduce computational costs while maintaining high relevance to large-scale benchmarks like MTEB.
- **Scaling Effect**: Accurately reflects scaling effects within the same model family and trends observed in top-performing models like NV-embed-v2.

## RAG Performance

- **Realistic Setup**: Replaces mixed context with top-5 chunks retrieved by the actual retriever, ensuring that performance always falls between base and oracle setups.

## Environment Setup

1. **Create Conda Environment**:
    ```sh
    conda create -n mirage python==3.11.11
    conda activate mirage
    ```
2. **Clone Repository**:
    ```sh
    git clone https://github.com/JohnnyNLP/MIRAGE.git
    cd MIRAGE
    ```
3. **Install Requirements**:
    ```sh
    pip install -r requirements.txt
    ```
4. **Run Main Script**:
    ```sh
    python main.py
    ```
5. **Modify Configuration** (if needed):
    - Edit `config.yaml` as required.

6. **Run Evaluation Script**:
    ```sh
    python evaluation.py
    ```

## File Descriptions

### config.yaml
- Contains default settings for 4 LLMs and 5 retrievers used in the main experiments.
- Designed to run on a single GPU (A6000).

### main.py
- Supports three modes: RAG, LLM, RET.
- Configurable arguments via `config.yaml`.
- Uses vLLM for LLM inference and SentenceTransformer for retriever inference.
- Default setup is 5-shot, balancing retrieval pool size and optimal RAG performance.

### evaluation.py
- Evaluates retriever, LLM, and RAG performance.
- LLMs are evaluated using EM Score, retrievers using F1, NDCG, and Acc, and RAG performance using four metrics proposed in the MIRAGE paper.
- The detailed report can be found in the `Evaluation_result` directory:
    - `LLM_result.jsonl` shows F1, EM_loose, and EM_strict scores. (Note that EM_loose score is more reliable than EM_strict since LLM tends to generate verbose responses)
    - `RET_result.jsonl` shows F1, NDCG, precision, and recall scores at 1, 3, and 5 respectively.
    - `Metrics.jsonl` shows 4 MIRAGE metrics scores: noise vulnerability, context acceptability, context insensitivity, and context misinterpretation.
    - When running the script, you can also see the ranking and overall score of each system.
    - The overall score is calculated as `-NV + CA - CI - CM`.

## Application

- **Simple and Fast**: Designed for quick and easy use with minimal computational resources.
- **Effective for LLM/Retriever/RAG Experiments**: Provides datasets and code for effective experimentation without heavy resource requirements.
- **Hugging Face Integration**: You can find our MIRAGE dataest at huggingface now!
    ```sh
    load_dataset('nlpai-lab/mirage')
    ```

## Notes

- **vLLM Framework**: Supports multi-GPU inference for LLMs.
- **Single GPU for Retriever**: Currently supports single GPU inference for retrievers using SentenceTransformer.
- **Batch API for Cost Efficiency**: Consider using batch API to reduce costs, especially for GPT-4o inference.

## Cost Considerations

- **GPT-4o Inference**: Costs approximately $70 for a single run. This may be subjected to change depending on the openAI's price policy.
- **Batch API**: Recommended for cost savings.

## Citation

```text
@misc{park2025miragemetricintensivebenchmarkretrievalaugmented,
      title={MIRAGE: A Metric-Intensive Benchmark for Retrieval-Augmented Generation Evaluation}, 
      author={Chanhee Park and Hyeonseok Moon and Chanjun Park and Heuiseok Lim},
      year={2025},
      eprint={2504.17137},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.17137}, 
}
```

