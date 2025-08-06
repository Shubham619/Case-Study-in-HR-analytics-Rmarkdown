
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"


export CMAKE_ARGS="\
-DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
-DUSE_CUDA=ON \
-DUSE_SYSTEM_NVTX=ON \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_PREFIX_PATH=$CUDA_HOME"

sudo python3 setup.py clean
sudo python3 setup.py install
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with your actual model name (e.g., 'deepseek-ai/deepseek-llm-7b-base')
model_name = "deepseek-ai/deepseek-llm-7b-base"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

def inspect_cache(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output = model(input_ids, use_cache=True)
        past_key_values = output.past_key_values  # Tuple of layers -> tuple(key, value)

    print(f"Prompt token length: {input_ids.shape[1]}")
    total = 0
    for l, layer in enumerate(past_key_values):
        for t in layer:
            size = t.element_size() * t.numel()
            total += size
        print(f"Layer {l} Key shape: {layer[0].shape}, Value shape: {layer[1].shape}")
    print(f"Total KV Cache Size: {total / 1024 / 1024:.2f} MB\n")

# Test with increasing prompt length
for n in [5, 10, 20, 40]:
    prompt = "Hello world! " * n
    inspect_cache(prompt)
