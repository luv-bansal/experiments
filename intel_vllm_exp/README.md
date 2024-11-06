## Intel vLLM experiments

### Repository Files

- **Dockerfile.vllm**: Dockerfile for creating a container with vLLM installed.
- **vllm_server.sh**: Script to start the vLLM OpenAI-compatible server.
- **vllm_server_inference.py**: Client script for performing inference requests to the vLLM server.
- **vllm_benchmark_locust.py**: Benchmarking script for evaluating vLLM performance.


### Benchmarking parameters

#### vLLM serving parameter
- gpu-memory-utilization: 0.9

#### vLLM config parameters
- input tokens: 10
- output tokens: 100


### Benchmarking Results

### Model: meta-llama/Meta-Llama-3-8B-Instruct

|**Batch Request**|   |**Nvidia H100**|**Nvidia A100**|**Nvidia A10**|**Intel Gaudi 2**|
| --- | --- | --- | --- | --- | --- |
| 1 |   | 105.72 tokens/s | 59.50 tokens/s | 26.59 tokens/s | 60.05 tokens/s |
| 10 |   | 501.04 tokens/s | 394.91 tokens/s | 113.71 tokens/s | 376.41 tokens/s |
| 100 |   | 999.7 tokens/s | 559.7 tokens/s | 1331.4 tokens/s | 779.2 tokens/s |
| 256 |   | 5113.5 tokens/s | 2628.3 tokens/s | 1747.8 tokens/s | 1267.9 tokens/s |