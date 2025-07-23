from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import modeling_utils
import torch
import os
os.environ["FLASH_ATTENTION_FORCE_DISABLED"] = "1"

tokenizer=AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model= AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.float32,
    max_memory={
        0:'5GB',
        1:"5GB",
        'cpu':"16GB"
    },
    offload_folder="/offload_nvm",
    offload_state_dict=True,
    
    )
question=input("What is your Question ::")
message=[[
    {
        "role":"system",
        "content":"You are a helpful assistant help user in their query"
    },
    {
        "role":"user",
        "content":question
    }
],
         [
    {
        "role":"system",
        "content":"You are a helpful assistant help user in their query"
    },
    {
        "role":"user",
        "content":question
    }
]]

prompt=[tokenizer.apply_chat_template(m,tokenize=False) for m in message]
input_id=tokenizer(prompt,return_tensors='pt')
output=model.generate(**input_id,max_new_tokens=100)
print(tokenizer.decode(output[0],skip_special_tokens=True))

