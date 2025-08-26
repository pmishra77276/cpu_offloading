from vllm import LLM, SamplingParams
import time
# Load a Hugging Face LLM (e.g., LLaMA, Falcon, Mistral)
from vllm import LLM
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    max_model_len=8192,   # instead of 131072
    gpu_memory_utilization=0.9,
    enforce_eager=True
)


# Define generation parameters
params = SamplingParams(temperature=0, top_p=0, max_tokens=100)

# Run inference
start=time.time()
print('start')
outputs = llm.generate(["Hello, how are you?"], params)
print(time.time()-start)
for output in outputs:
    print(output.outputs[0].text)
