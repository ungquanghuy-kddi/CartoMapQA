
def parse_gpt4(response):
    response = response.strip()
    answer = response[0]
    if answer not in ['A', 'B', 'C', 'D']:
        answer = ""
    return answer

def parse_gemini(response):
    response = response.strip()
    answer = response[0]
    if answer not in ['A', 'B', 'C', 'D']:
        answer = ""
    return answer

def parse_qwen_vl(response):
    response = response.strip()
    if "correct answer is " in response:
        tmp = response.split('correct answer is ')[1]
        tmp = tmp.strip().split(' ')[0].replace('.', '')
        if tmp in ['A', 'B', 'C', 'D']:
            answer = tmp
        else:
            answer = "" 
    else:
        answer = response[0]
        if answer not in ['A', 'B', 'C', 'D']:
            answer = ""
    return answer

def parse_llava(response):
    answer = response.strip()[0]
    if answer not in ['A', 'B', 'C', 'D']:
        answer = ""
    return answer

def parse_hpt(response):
    response = response.strip()
    answer = response[0]
    if answer not in ['A', 'B', 'C', 'D']:
        answer = ""
    return answer

def parse_internvl(response):
    response = response.strip()
    answer = response[0]
    if answer not in ['A', 'B', 'C', 'D']:
        answer = ""
    return answer

def parse_cogvlm(response):
    response = response.strip()
    if "correct answer is " in response:
        tmp = response.split('correct answer is ')[1]
        tmp = tmp.strip().split(' ')[0].replace('.', '')
        if tmp in ['A', 'B', 'C', 'D']:
            answer = tmp
        else:
            answer = "" 
    else:
        answer = response[0]
        if answer not in ['A', 'B', 'C', 'D']:
            answer = ""
    return answer

def parse_vila(response):
    response = response.strip()
    answer = response[0]
    if answer not in ['A', 'B', 'C', 'D']:
        answer = ""
    return answer