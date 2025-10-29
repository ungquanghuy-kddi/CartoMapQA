
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

from Prompts.prompt_system import system_prompt

def init_llama4_scout(model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct"):
    processor = AutoProcessor.from_pretrained(model_id)
    model = Llama4ForConditionalGeneration.from_pretrained(
        model_id,
        attn_implementation="eager",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return model, processor

def get_response_llama4(model, prompt, image_path, processor, max_tokens=512):
    messages = [
    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    {"role": "user", "content": [
        {"type": "image", "path": image_path},
        {"type": "text", "text": prompt}
    ]}
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
    )

    response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
    return response