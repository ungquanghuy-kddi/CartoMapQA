import json
from utils import  text2int
import numpy as np
import re

TYPE = {
    "presence": 1,
    "counting": 2,
    "name_listing": 3,
}

def parse_gpt4(response, type):
    response = response.strip()
    type = TYPE[type]
    if type == 1:
        tmp = response.split("\n")[0].lower().translate(str.maketrans('','', "-,.")).strip()
        if tmp.startswith('yes'):
            pred = 'yes'
        elif tmp.startswith('no'):
            pred = 'no'
        else:
            # random answer
            print ("Randomness in type 1")
            pred = 'yes' if np.random.randint(0, 2) == 1 else 'no'

    if type == 2:
        tmp = response.split("\n")[0].strip()
        if tmp.isnumeric():
            pred = int(tmp)
        elif text2int(tmp) != -1:
            pred = int(text2int(tmp))
        else:
            # random answer
            print ("Randomness in type 2")
            pred = np.random.randint(0, 17)

    if type == 3:
        pred = [item.strip() for item in response.strip().split('\n')]

    return pred

def parse_gemini(response, type):
    response = response.strip()
    type = TYPE[type]
    if type == 1:
        tmp = response.split("\n")[0].lower().strip().translate(str.maketrans('','', "-,."))
        if tmp.startswith('yes'):
            pred = 'yes'
        elif tmp.startswith('no'):
            pred = 'no'
        else:
            # random answer
            print ("Randomness in type 1")
            pred = 'yes' if np.random.randint(0, 2) == 1 else 'no'

    if type == 2:
        tmp = response.split("\n")[0].strip().translate(str.maketrans('','', "-,."))
        if not tmp.isnumeric():
            if text2int(tmp) != -1:
                pred = text2int(tmp)
            else:
                # random answer
                print ("Randomness in type 2")
                pred = np.random.randint(0, 17)
        else:
            pred = int(tmp)

    if type == 3:
        pred = response.strip().split('\n')
    return pred

def parse_qwen_vl(response, type):
    response = response.strip()
    type = TYPE[type]
    if type == 1:
        tmp = response.strip().lower().translate(str.maketrans('','', "-,."))
        if tmp.startswith('yes'):
            pred = 'yes'
        elif tmp.startswith('no'):
            pred = 'no'
        else:
            # random answer
            print ("Randomness in type 1")
            pred = 'yes' if np.random.randint(0, 2) == 1 else 'no'

    if type == 2:
        tmp = response.split("\n")[0].translate(str.maketrans('','', "-,."))
        if tmp.isnumeric():
            pred = int(tmp)
        else:
            if "There are " in response:
                tmp = response.strip().split("There are ")[1]
            elif "there are " in response:
                tmp = response.strip().split("there are ")[1]
            elif "There is " in response:
                tmp = response.strip().split("There is ")[1]
            elif "there is " in response:
                tmp = response.strip().split("there is ")[1]
            
            tmp = tmp.split(" ")[0].strip()
            if tmp.isnumeric():
                pred = int(tmp)
            elif text2int(tmp) != -1:
                pred = int(text2int(tmp))                 
            else:
                print ("Randomness in type 2")
                pred = np.random.randint(0, 17)

    if type == 3:
        # read response in json format
        response = response.replace("json", "")
        response = response.replace("`", "")
        parsed = json.loads(response)
        pred = list(parsed.values())

    return pred

def parse_llava(response, type):
    response = response.strip()
    type = TYPE[type]
    if type == 1:
        tmp = response.split(",")[0].strip().lower()
        if tmp.startswith('yes'):
            pred = 'yes'
        elif tmp.startswith('no'):
            pred = 'no'
        else:
            # random answer
            print ("Randomness in type 1")
            pred = 'yes' if np.random.randint(0, 2) == 1 else 'no'

    if type == 2:
        tmp = response.strip().split("\n")[0]
        if tmp.isnumeric():
            pred = int(tmp)
        else:
            tmp = response.strip().split("There are ")[1]
            tmp = tmp.split(" ")[0].strip()
            if tmp.isnumeric():
                pred = int(tmp)
            elif tmp == "no":
                pred = 0
            elif text2int(tmp) != -1:
                pred = int(text2int(tmp))                   
            else:
                print ("Randomness in type 2")
                pred = np.random.randint(0, 17)

    if type == 3:
        # matching pattern Numbered list
        pattern = r"\d+\.\s(.+)"
        parsed = re.findall(pattern, response)
        if len(parsed) > 0:
            pred = parsed
        else:
            pred = []
    return pred

def parse_hpt(response, type):
    response = response.strip()
    type = TYPE[type]
    if type == 1:
        tmp = response.split("\n")[0].lower()
        tmp = tmp.translate(str.maketrans('','', "-,.")) # delete dot

        if tmp.startswith('yes'):
            pred = 'yes'
        elif tmp.startswith('no'):
            pred = 'no'
        else:
            # random answer
            print ("Randomness in type 1")
            pred = 'yes' if np.random.randint(0, 2) == 1 else 'no'

    if type == 2:
        tmp = response.strip().split("\n")[0]
        tmp = tmp.replace('.', '')  # delete dot
        tmp = tmp.replace(',', '')
        if tmp.isnumeric():
            pred = int(tmp)
        else:
            if "There are " in tmp:
                tmp = response.strip().split("There are ")[1]
            tmp = tmp.split(" ")[0].strip()
            if tmp.isnumeric():
                pred = int(tmp)
            elif tmp == "no":
                pred = 0
            elif text2int(tmp) != -1:
                pred = int(text2int(tmp))                   
            else:
                print ("Randomness in type 2")
                pred = np.random.randint(0, 17)

    if type == 3:
        pred = response.replace('.', '').split("\n")

    return pred

def parse_internvl(response, type):
    response = response.strip()
    type = TYPE[type]
    if type == 1:
        tmp = response.split("\n")[0].lower().translate(str.maketrans('','', "-,."))
        if tmp.startswith('yes'):
            pred = 'yes'
        elif tmp.startswith('no'):
            pred = 'no'
        else:
            # random answer
            print ("Randomness in type 1")
            pred = 'yes' if np.random.randint(0, 2) == 1 else 'no'

    if type == 2:
        tmp = response.split("\n")[0].translate(str.maketrans('','', "-,."))

        if tmp.isnumeric():
            pred = int(tmp)
        else:
            tmp = response.lower()
            if "there are " in tmp:
                tmp = tmp.strip().split("there are ")[1]
            elif "there is " in tmp:
                tmp = tmp.strip().split("there is ")[1]

            tmp = tmp.split(" ")[0].strip()
            if tmp.isnumeric():
                pred = int(tmp)
            elif tmp == "no":
                pred = 0
            elif text2int(tmp) != -1:
                pred = int(text2int(tmp))                   
            else:
                print ("Randomness in type 2")
                # print (response)
                # print (tmp)
                print (response.lower().strip().split("there are ")[1]) 
                pred = np.random.randint(0, 17)

    if type == 3:
        response = response.replace("json", "")
        response = response.replace("`", "")
        parsed = json.loads(response)
        pred = list(parsed.values())
    return pred

def parse_cogvlm(response, type):
    response = response.strip()
    type = TYPE[type]
    if type == 1:
        tmp = response.strip().lower().translate(str.maketrans('','', "-,."))
        if tmp.startswith('yes'):
            pred = 'yes'
        elif tmp.startswith('no'):
            pred = 'no'
        else:
            # random answer
            print ("Randomness in type 1")
            pred = 'yes' if np.random.randint(0, 2) == 1 else 'no'

    if type == 2:
        tmp = response.strip().split("\n")[0].translate(str.maketrans('','', "-,."))

        if tmp.isnumeric():
            pred = int(tmp)
        else:
            tmp = response.lower()
            if "there are " in tmp:
                tmp = tmp.strip().split("there are ")[1]
            elif "there is " in tmp:
                tmp = tmp.strip().split("there is ")[1]

            tmp = tmp.split(" ")[0].strip()
            if tmp.isnumeric():
                pred = int(tmp)
            elif tmp == "no":
                pred = 0
            elif text2int(tmp) != -1:
                pred = int(text2int(tmp))                   
            else:
                print ("Randomness in type 2")
                pred = np.random.randint(0, 17)

    if type == 3:
        response = response.replace("json", "")
        response = response.replace("`", "")
        parsed = json.loads(response)
        pred = list(parsed.values())
        if isinstance(pred[0], list):
            pred = pred[0]
    return pred

def parse_vila(response, type):
    response = response.strip()
    type = TYPE[type]
    if type == 1:
        tmp = response.strip().lower().translate(str.maketrans('','', "-,."))
        if tmp.startswith('yes'):
            pred = 'yes'
        elif tmp.startswith('no'):
            pred = 'no'
        else:
            # random answer
            print ("Randomness in type 1")
            pred = 'yes' if np.random.randint(0, 2) == 1 else 'no'

    if type == 2:
        tmp = response.strip().split("\n")[0].translate(str.maketrans('','', "-,."))

        if tmp.isnumeric():
            pred = int(tmp)
        else:
            tmp = response.lower()
            if "there are " in tmp:
                tmp = tmp.strip().split("there are ")[1]
            
            if "there is " in tmp:
                tmp = tmp.strip().split("there is ")[1]

            tmp = tmp.split(" ")[0].strip()
            if tmp.isnumeric():
                pred = int(tmp)
            elif tmp == "no":
                pred = 0
            elif text2int(tmp) != -1:
                pred = int(text2int(tmp))                   
            else:
                print ("Randomness in type 2")
                pred = np.random.randint(0, 17)

    if type == 3:
        # matching pattern Numbered list
        pattern = r"\d+\.\s(.+)"
        parsed = re.findall(pattern, response)
        if len(parsed) > 0:
            pred = parsed
        else:
            pred = response.strip().split('\n')
    return pred