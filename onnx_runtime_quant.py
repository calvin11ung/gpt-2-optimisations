import torch
from onnxruntime.transformers.models.gpt2.gpt2_helper import GPT2LMHeadModel
from transformers import AutoConfig
import os
from onnxruntime.transformers.quantize_helper import QuantizeHelper

cache_dir = os.path.join(".", "cache_models")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

model_name_or_path = "gpt2-xl"
config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
device = torch.device("cpu")
model.eval().to(device)

model = QuantizeHelper.quantize_torch_model(model)

torch.save(model, 'gpt2-xl-quantized.pt')