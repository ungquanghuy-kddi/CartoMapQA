from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

def init_qwen_vl():
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    # use cuda device
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    return model, tokenizer

def get_response(model, tokenizer, prompt, image_path):
    # 1st dialogue turn
    query = tokenizer.from_list_format([
        {'image': image_path}, # Either a local path or an url
        {'text': prompt},
    ])
    out, _ = model.chat(tokenizer, query=query, history=None)

    return out