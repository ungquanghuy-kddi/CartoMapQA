from eval import normalize_route

def parse_srn(response):
    response = response.strip()
    response = response.replace("**Answer:**", "") if "**Answer:**" in response else response
    response = response.replace("*", "")
    response = response.replace("Answer:", "")
    response = response.replace("[", "").replace("]", "")
    response = response.replace("<|eot|>", "")
    response = response.split(",")
    response = [item.strip() for item in response]
    norm_response = normalize_route(response)
    return norm_response