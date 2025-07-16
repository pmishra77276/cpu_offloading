from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
device='cuda'
model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
tokenizer=AutoTokenizer.from_pretrained(model_name,)
if tokenizer.pad_token==None:
    tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(model_name,
                                           torch_dtype=torch.bfloat16).to(device)

message=[
    {
        "role":"system",
        "content":"you are a my sql query generater. Generate query based on user prompt"
    },
    {
        "role":"user",
        "content":"i want to see how many people were absent yesterday [table:- employees,metadata: 'emp id','leaves','working hrs']"
    }
]


prompt=tokenizer.apply_chat_template(message, tokenize=False)
inputs=tokenizer(prompt,return_tensors='pt').to(device)

out=model.generate(inputs['input_ids'],max_length=500)

print(tokenizer.decode(out[0]))
