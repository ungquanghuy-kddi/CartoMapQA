import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PIL import Image


def init_cogvlm2(MODEL_PATH="THUDM/cogvlm2-llama3-chat-19B", device='cuda'):
    if torch.cuda.is_available() == False: 
        device = 'cpu'
        print ("Use CPU!")    
    
    torch_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
        0] >= 8 else torch.float16

    # Argument parser
    quant = 8 # [4, 8]

    if 'int4' in MODEL_PATH:
        quant = 4

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    # Check GPU memory
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 48 * 1024 ** 3 and not quant:
        print("GPU memory is less than 48GB. Please use cli_demo_multi_gpus.py or pass `--quant 4` or `--quant 8`.")
        exit()

    # Load the model
    if quant == 4:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_type,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            low_cpu_mem_usage=True
        ).eval()
    elif quant == 8:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_type,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            low_cpu_mem_usage=True
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_type,
            trust_remote_code=True
        ).eval().to(device)
    
    return model, tokenizer

def get_response(model, tokenizer, prompt, image_path, device='cuda'):
    if torch.cuda.is_available() == False: 
        device = 'cpu'
        print ("Use CPU!")
        
    torch_type = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
        0] >= 8 else torch.float16
    image = Image.open(image_path).convert('RGB')

    input_by_model = model.build_conversation_input_ids(
        tokenizer,
        query=prompt,
        history=[],
        images=[image],
        template_version='chat'
        )

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
        'images': [[input_by_model['images'][0].to(device).to(torch_type)]] if image is not None else None,
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        out = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return out