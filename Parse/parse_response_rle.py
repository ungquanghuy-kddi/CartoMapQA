from utils import is_float, find_indices

def parse_rle(response):
    tmp = response.strip().split("\n")[0].replace("-", "").replace(",", "").replace("*", "").strip()
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
            if "which converts to" in response:
                pred = response.strip().split("which converts to ")[1].strip()
                pred = pred.split(" ")[0]
            else:
                pred = -1
    return pred