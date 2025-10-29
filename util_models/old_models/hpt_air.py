from vlmeval.config import supported_VLM

def init_hpt_air():
    model_name = "hpt-air-1-5"
    model = supported_VLM[model_name]() 
    return model

def get_response(model, prompt, image_path):
    out = model.generate(prompt=prompt, image_path=image_path, dataset='demo')
    return out