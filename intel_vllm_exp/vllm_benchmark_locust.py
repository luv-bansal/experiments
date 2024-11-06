import os
import time
from locust import HttpUser, task, events, between
import requests
from openai import OpenAI

concurrent_requests= 100

class VLLMUser(HttpUser):
    wait_time = between(1, 5)  # Simulate a user waiting between 1 and 5 seconds between tasks
    total_requests = 0
    total_failures = 0
    start_time = 0
    user_counter = 0
    request_count = 0
    total_tokens = 0
    response_times = []
    test_start_time = None
    stop_flag = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        VLLMUser.user_counter += 1
        self.user_id = VLLMUser.user_counter

    def on_start(self):
        print(f"*********** on_start called for User {self.user_id} **********")
        self.openai_api_base = self.host + "/v1"
        self.openai_api_key = "API_KEY"
        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )
        self.start_time = time.time()
        
        # Initialize test start time if it's the first user
        if VLLMUser.test_start_time is None:
            VLLMUser.test_start_time = time.time()

    def on_stop(self):
        print(f"*********** on_stop called for User {self.user_id} **********")
        
        end_time = time.time()
        total_time = end_time - self.start_time
        if total_time > 0:
            # Increment the global request count
            VLLMUser.request_count += self.total_requests
            print(f"Total Requests by User {self.user_id}: {self.total_requests}")
            print(f"Total Failures by User {self.user_id}: {self.total_failures}")
            print(f"Total Time for User {self.user_id}: {total_time:.2f} seconds")

        # Check if all users have completed their requests
        if VLLMUser.user_counter == concurrent_requests and not VLLMUser.stop_flag:
            VLLMUser.stop_flag = True
            # Trigger custom quitting logic
            self.environment.process_exit_code = 0
            self.environment.runner.quit()
            
    @task
    def load_test(self):
        if self.total_requests > 0:
            return  # Ensure only one request per user

        prompts = ["Write a long paragraph about India"]
        mkwargs = {
            'temperature': 0.7,
            'max_tokens': 100,
            'top_p': 0.9
        }
        max_tokens = mkwargs.get('max_tokens')
        start_time = time.time()
        try:
            stream_text = self.vllm_inference(prompts, **mkwargs)

            first_time = 0
            st = start_time
            time_duration = []
            tokens_generated = 0
            for i, text in enumerate(stream_text):
                if i == 0:
                    first_time = time.time() - st
                tokens_generated += len(text.split())
                time_duration.append(time.time() - st)
                st = time.time()

            end_time = time.time()
            processing_time = end_time - start_time

            # Record response time
            VLLMUser.response_times.append(processing_time)

            # Logging the results
            events.request.fire(
                request_type="vLLM Inference",
                name="vLLM Load Test",
                response_time=processing_time,  # Convert to milliseconds
                response_length=max_tokens,  # Adjust as needed
            )
            # Increment total requests and tokens for this user
            self.total_requests += 1
            VLLMUser.total_tokens += tokens_generated

        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time

            events.request.fire(
                request_type="vLLM Inference",
                name="vLLM Load Test",
                response_time=processing_time,
                exception=e,
            )
            # Increment total requests, failures and tokens for this user
            self.total_requests += 1
            VLLMUser.total_failures += 1

        # Stop user after a single request
        self.environment.runner.quit()

    def vllm_inference(self, prompts, **mkwargs):
        messages = [{"role": "user", "content": prompts[0]}]
        temperature = mkwargs.get('temperature')
        max_tokens = mkwargs.get('max_tokens')
        top_p = mkwargs.get('top_p')

        model = 'meta-llama/Meta-Llama-3-8B-Instruct'

        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
        )
        response = self.client.chat.completions.create(**kwargs)
        
        for chunk in response:
            if chunk.choices[0].delta.content is None:
                continue
            yield chunk.choices[0].delta.content

# Hook to calculate final metrics after all users stop
@events.quitting.add_listener
def on_quitting(environment, **_kwargs):
    end_time = time.time()
    total_duration = end_time - VLLMUser.test_start_time
    if total_duration > 0:
        rps = VLLMUser.request_count / total_duration
        throughput = VLLMUser.total_tokens / total_duration
        average_response_time = sum(VLLMUser.response_times) / len(VLLMUser.response_times) if VLLMUser.response_times else 0
        print(f"Total Requests: {VLLMUser.request_count}")
        print(f"Total Failures: {VLLMUser.total_failures}")
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Requests per Second (RPS): {rps:.2f}")
        print(f"Throughput (Tokens per Second): {throughput:.2f}")
        print(f"Average Response Time: {average_response_time:.2f} seconds")

if __name__ == "__main__":
    import os
    os.system(f"locust -f locust_vllm.py --headless -u {concurrent_requests} -r {concurrent_requests} --host=http://localhost:7000 --logfile=vllm_load_testing.log --csv=vllm_load_testing")
