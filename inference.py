import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
tok=AutoTokenizer.from_pretrained("facebook/opt-125m")
model=AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
question="what are you doing?"
prompt=tok(question,return_tensors='pt').to('cuda')

input_ids=prompt['input_ids']
del prompt
torch.cuda.empty_cache()

model.model.decoder.embed_tokens.to('cuda')
x = model.model.decoder.embed_tokens(input_ids)
input_ids.to('cpu')
del input_ids
torch.cuda.empty_cache()

# Inputs to model

model.model.decoder.embed_tokens.to('cpu')
torch.cuda.empty_cache()
model.model.decoder.embed_positions.to('cuda')
x = x + model.model.decoder.embed_positions(torch.arange(x.shape[1]).unsqueeze(0).to('cuda'))
model.model.decoder.embed_positions.to('cpu')
torch.cuda.empty_cache()
print()
cnt=0
for i in model.model.decoder.layers:
    cnt+=1
    print(cnt)
    try:
        i.to('cuda')
        x.to('cuda')
        with torch.no_grad():
            x=i(x)[0]
        i.to('cpu')
        torch.cuda.empty_cache()
        x.to('cpu')
        torch.cuda.empty_cache()
    except:
        print(cnt)
        print(x)
        break
model.lm_head.to('cuda')
x=model.lm_head(x.to('cuda'))
model.lm_head.to('cpu')
torch.cuda.empty_cache()
x1=x.argmax(-1)
tok.decode(x1[0])

