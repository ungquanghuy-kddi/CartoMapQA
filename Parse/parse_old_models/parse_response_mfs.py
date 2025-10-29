from utils import is_float, find_indices

def parse_gpt4(response):
    tmp = response.split("\n")[0].lower()
    tmp = tmp.replace('about', '').replace('approximately', '').replace(',', '').strip()
    pred = tmp.split(" ")[0]

    if not is_float(pred): # invalid result
        pred = -1
    return pred

def parse_gemini(response):
    tmp = response.split("\n")[0]
    if " is " in tmp:
        tmp = tmp.split(" is ")[1].strip()
        tmp = tmp.replace('about', '').replace('approximately', '').replace(',', '').strip()
        pred = tmp.split(" ")[0]
    else:
        ### search for a number in the first line
        tmp = tmp.split(" ")
        find_num = find_indices(tmp, lambda e: is_float(e))
        if len(find_num) > 0:
            pred = tmp[find_num[0]]
        else:
            pred = -1
    return pred

def parse_qwen_vl(response):
    tmp = response.replace(',', '').replace('m', '')
    tmp = tmp.split(" ")
    find_num = find_indices(tmp, lambda e: is_float(e))
    if len(find_num) > 0:
        pred = tmp[find_num[0]]
    else:
        pred = -1

    return pred

def parse_llava(response):
    tmp = response.strip().replace(',', '').replace('\n', '')

    ### search for a number in the first line
    tmp = tmp.split(" ")
    find_num = find_indices(tmp, lambda e: is_float(e))
    if len(find_num) > 0:
        pred = tmp[find_num[-1]]
    else:
        pred = -1
    return pred

def parse_hpt(response):
    tmp = response.replace(',', '').split("\n")[0]
    if " is " in tmp:
        tmp = tmp.split(" is ")[1].strip()
        tmp = tmp.replace('about', '').replace('approximately', '').replace(',', '').strip()
        pred = tmp.split(" ")[0]
    else:
        ### search for a number in the first line
        tmp = tmp.split(" ")
        find_num = find_indices(tmp, lambda e: is_float(e))
        if len(find_num) > 0:
            pred = tmp[find_num[0]]
        else:
            pred = -1

    return pred

def parse_internvl(response):
    tmp = response.split("\n")[0].replace("*", "")
    if " is " in tmp:
        tmp = tmp.split(" is ")[1].strip()
        tmp = tmp.replace('about', '').replace('approximately', '').replace(',', '').strip()
        pred = tmp.split(" ")[0]
    else:
        ### search for a number in response
        tmp = response.replace("*", "").split(" ")
        find_num = find_indices(tmp, lambda e: is_float(e))
        if len(find_num) > 0:
            pred = tmp[find_num[0]]
        else:
            pred = -1
    
    return pred

def parse_cogvlm(response):
    tmp = response.split("\n")[0]
    if " is " in tmp:
        tmp = tmp.split(" is ")[1].strip()
        tmp = tmp.replace('about', '').replace('approximately', '').replace(',', '').strip()
        pred = tmp.split(" ")[0]
    else:
        ### search for a number in the response
        tmp = response.split(" ")
        find_num = find_indices(tmp, lambda e: is_float(e))
        if len(find_num) > 0:
            pred = tmp[find_num[0]]
        else:
            pred = -1
    return pred

def parse_vila(response):
    ### search for a number in response
    tmp = response.replace(",", "").replace(".", "").split(" ")
    find_num = find_indices(tmp, lambda e: is_float(e))
    if len(find_num) > 0:
        pred = tmp[find_num[-1]]
    else:
        pred = -1
    return pred
