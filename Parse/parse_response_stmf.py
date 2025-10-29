from Prompts.prompt_stmf import TYPE
from utils import  text2int
import numpy as np
import re

def parse_stmf(response, type, fn):
    response = response.strip()
    type = TYPE[type]
    if type == 1:
        response = response.replace("**Answer:**", "") if "**Answer:**" in response else response
        response = response.replace("*", "")
        tmp = response.split("\n")[0].lower().translate(str.maketrans('','', "-,.")).strip()
        if tmp.startswith('yes'):
            pred = 'yes'
        elif tmp.startswith('no'):
            pred = 'no'
        else:
            # random answer
            print ("Randomness in type 1", fn)
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
                print ("Randomness in type 2", fn)
                pred = np.random.randint(0, 17)
    if type == 3:
        pred = [re.sub(r'^\d+\.\s', '', item).replace("*", "").strip() for item in response.strip().split('\n')]

    return pred
