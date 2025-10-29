import json

def parse_mml(response):
    response = response.strip()
    response = response.replace('json', '').replace('`', '').replace("<|eot|>", "")
    # Check if the response is a valid JSON string
    try:
        json_response = json.loads(response)
    except json.JSONDecodeError:
        # If not, return an empty dictionary or handle the error as needed
        print("Invalid JSON response:", response)
        return None
    return json_response