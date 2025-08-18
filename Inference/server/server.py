from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.responses import StreamingResponse
from inference import AccelerateOffloadInference
from functools import lru_cache
import asyncio
# ---------------------------
# Initialize FastAPI app
# ---------------------------
app = FastAPI(title="LLM Inference API")

print("🚀 Booting inference engine...")

# ---------------------------
# Load Inference Engine
# ---------------------------
# inference = AccelerateOffloadInference(
#     nvme_path="/offload_nvm",
#     max_gpu1_memory="7GB",
#     max_gpu2_memory="0GB",
#     max_cpu_memory="30GB",
#     model_name="meta-llama/Llama-3.2-3B-Instruct",
#     max_new_tokens=512,
#     temperature=0.7,
# )
# inference.load_model()


# ---------------------------
# Request Schema
# ---------------------------
class PromptRequest(BaseModel):
    prompt: str
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True

    # New params (all optional)
    model: str | None = None
    max_gpu1_memory: str | None = None
    max_gpu2_memory: str | None = None
    max_cpu_memory: str | None = None
    nvme_path: str | None = None
    max_new_tokens: int | None = None

model_cache = {}

def get_inference(config):
    key = tuple(config.items())
    if key not in model_cache:
        inf = AccelerateOffloadInference(**config)
        inf.load_model()
        model_cache[key] = inf
    return model_cache[key]

# ---------------------------
# API Routes
# ---------------------------
@app.post("/infer")
# @app.post("/infer")
async def infer(request: PromptRequest):
    config = {
        "nvme_path": request.nvme_path or "/offload_nvm",
        "max_gpu1_memory": request.max_gpu1_memory or "3GB",
        "max_gpu2_memory": request.max_gpu2_memory or "0GB",
        "max_cpu_memory": request.max_cpu_memory or "30GB",
        "model_name": request.model or "meta-llama/Llama-3.2-3B-Instruct",
        "max_new_tokens": request.max_new_tokens or 512,
        "temperature": 0.7,
    }
    inference = get_inference(config)

    response, gen_time = inference.generate_response(
        prompt=request.prompt,
        top_p=request.top_p,
        top_k=request.top_k,
        do_sample=request.do_sample,
    )
    return {"response": response, "generation_time": gen_time}
# async def infer(request: PromptRequest):
#     # If model/config passed → reload dynamically
#     global inference
#     if request.model or request.max_new_tokens or request.max_gpu1_memory:
#         inference = AccelerateOffloadInference(
#             nvme_path=request.nvme_path or "/offload_nvm",
#             max_gpu1_memory=request.max_gpu1_memory or "7GB",
#             max_gpu2_memory=request.max_gpu2_memory or "0GB",
#             max_cpu_memory=request.max_cpu_memory or "30GB",
#             model_name=request.model or "meta-llama/Llama-3.2-3B-Instruct",
#             max_new_tokens=request.max_new_tokens or 512,
#             temperature=0.7,
#         )
#         inference.load_model()
#       # so we don’t lose the instance

#     response, gen_time = inference.generate_response(
#         prompt=request.prompt,
#         top_p=request.top_p,
#         top_k=request.top_k,
#         do_sample=request.do_sample,
#     )
#     return {"response": response, "generation_time": gen_time}

@app.post("/infer_stream")
async def infer_stream(request: PromptRequest):
    config = {
        "nvme_path": request.nvme_path or "/offload_nvm",
        "max_gpu1_memory": request.max_gpu1_memory or "3GB",
        "max_gpu2_memory": request.max_gpu2_memory or "0GB",
        "max_cpu_memory": request.max_cpu_memory or "30GB",
        "model_name": request.model or "meta-llama/Llama-3.2-3B-Instruct",
        "max_new_tokens": request.max_new_tokens or 512,
        "temperature": 0.7,
    }
    inference = get_inference(config)
    streamer = inference.generate_stream(
        prompt=request.prompt,
        top_p=request.top_p,
        top_k=request.top_k,
        do_sample=request.do_sample,
    )

    async def event_generator():
        loop = asyncio.get_event_loop()
        for token in streamer:  # HuggingFace streamer yields tokens one by one
            # Format as SSE so curl/browser prints immediately
            yield f"data: {token}\n\n"
            await asyncio.sleep(0)  # let event loop breathe
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------
# Run with Uvicorn
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1147)
