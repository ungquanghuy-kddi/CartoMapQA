import google.generativeai as genai
import PIL.Image

from API_keys.keys import GEMINI_KEY

genai.configure(api_key=GEMINI_KEY)

def init_gemini(model_name='gemini-2.5-pro-preview-03-25'):
    model = genai.GenerativeModel(model_name)
    return model

def get_response(model, prompt, image_path):
    img = PIL.Image.open(image_path)
    response = model.generate_content([prompt, img], 
                                        safety_settings={
                                        'HATE': 'BLOCK_NONE',
                                        'HARASSMENT': 'BLOCK_NONE',
                                        'SEXUAL' : 'BLOCK_NONE',
                                        'DANGEROUS' : 'BLOCK_NONE'
        })
    
    response.resolve()
    out = response.text

    return out
