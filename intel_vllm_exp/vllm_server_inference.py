import time
from openai import OpenAI

def vllm_infernece( prompts, **mkwargs):
    openai_api_base = "http://localhost:7000/v1"
    openai_api_key = "API_KEY"
    client = OpenAI(
      api_key=openai_api_key,
      base_url=openai_api_base,
    )
    messages = [{"role": "user", "content": prompts[0]}]

    # temperature = mkwargs.get('temperature', 0.7)
    max_tokens = mkwargs.get('max_tokens', 100)
    # top_p = mkwargs.get('top_p', 0.9)
    # min_tokens= mkwargs.get('min_tokens', 99)
    # top_k= mkwargs.get('top_k', 50)
    n= mkwargs.get('n', 1)

    models = client.models.list()
    model = models.data[0].id

    kwargs = dict(
        model=model,
        messages=messages,
        # temperature=temperature,
        max_tokens=max_tokens,
        # top_p=top_p,
        # min_tokens=min_tokens,
        # top_k=top_k,
        n=n,
        stream=True,
    )
    response = client.chat.completions.create(**kwargs)
    
    for chunk in response:
        if chunk.choices[0].delta.content is None:
            continue
        yield chunk.choices[0].delta.content





def main( prompt, **mkwargs ):

    max_tokens = mkwargs.get('max_tokens')

    start_time = time.time()
    stream_text= vllm_infernece(prompt, **mkwargs)


    print("\nGenerated Texts:")
    first_time=0
    st = start_time
    time_duration =[]
    for i, text in enumerate(stream_text):
        if i==0:
            first_time=time.time() - st
        time_duration.append(time.time() - st)
        st = time.time()
        print(text, end='', flush=True)


    # gpu_utilization = get_gpu_utilization()
    end_time = time.time()
    processing_time = end_time - start_time
    



    # Time latencies
    print(f"First token generation latency: {first_time:.2f} seconds")
    print(f"Average Time per response: {sum(time_duration) / len(time_duration):.2f} seconds")
    print(f"Total Processing Time: {processing_time:.2f} seconds")
    print(f"Output throuput in tokens per second: {(len(time_duration))/(processing_time):.2f} tokens /seconds")
    print(f"Total output tokens: {len(time_duration)}")

# Sample prompts.
# Sample prompts.
prompts = [
    "Write long paragraph about LLMs.",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
    # "tell me a n"
]


if __name__ == "__main__":
    gpu_memory_utilization=0.9
    main( prompts, max_tokens=100, n=1)