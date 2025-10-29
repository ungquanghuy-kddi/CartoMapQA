from Prompts.prompt_mtmf import POI_LIST, POI_ALTER_NAME

def parse_mtmf(response):
    norm_response = {}
    for item in response:
        # check in the list
        lc_item = item.lower()
        for k, cands in POI_ALTER_NAME.items():
            if lc_item in cands:
                norm_response[POI_LIST[k]] = response[item]
    return norm_response