
def text2int(textnum, numwords={}):
    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

      tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

      scales = ["hundred", "thousand", "million", "billion", "trillion"]

      numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.split():
        if word not in numwords:
          return -1
        
        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

import json

# Function to extract JSON from a mixed content file
def extract_json_from_file(content):
    # Tracking braces to find JSON content
    json_start = None
    brace_count = 0

    for index, char in enumerate(content):
        if char == '{':
            if json_start is None:
                json_start = index
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and json_start is not None:
                json_string = content[json_start:index + 1]
                try:
                    json_content = json.loads(json_string)
                    return json_string
                except json.JSONDecodeError:
                    print("Found JSON-like content, but it's not valid JSON.")
                return

    print("No JSON content found.")
    return -1