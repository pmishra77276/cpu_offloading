from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch

device='cuda'
model_name="mistralai/Mistral-7B-Instruct-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
tokenizer=AutoTokenizer.from_pretrained(model_name,)
if tokenizer.pad_token==None:
    tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(model_name,
                                           torch_dtype=torch.bfloat16,
                                           quantization_config=bnb_config).to(device)

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
