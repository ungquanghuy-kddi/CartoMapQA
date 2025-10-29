
POI_ALTER_NAME = {
    "0": ["convenience store", "convenience store"],
    "1": ["supermarket"],
    "2": ["drug store/pharmacy", "drug_store/pharmacy"],
    "3": ["general clinic", "general_clinic"],
    "4": ["hospital"],
    "5": ["restaurant"],
    "6": ["fast food store", "fast_food_store"],
    "7": ["coffee shop", "coffee_shop"],
    "8": ["atm or cash point", "atm_or_cash_point"],
    "9": ["bank"],
    "10": ["playground"],
    "11": ["area of park", "area_of_park"],
    "12": ["fitness center", "fitness_center"],
    "13": ["clothes shop", "clothes_shop"],
    "14": ["beauty shop", "beauty_shop"],
    "15": ["water stream", "water_stream"],
    "16": ["river"],
    "17": ["bus stop", "bus_stop"],
    "18": ["gas station", "gas_station"],
    "19": ["primary road", "primary_road"],
    "20": ["car parking place", "car_parking_place"],
    "21": ["public toilet", "public_toilet"],
}

POI_LIST = {
    "0": "convenience store",
    "1": "supermarket",
    "2": "drug store/pharmacy",
    "3": "clinic",
    "4": "hospital",
    "5": "restaurant",
    "6": "fast food store",
    "7": "coffee shop",
    "8": "ATM or cash point",
    "9": "bank",
    "10": "playground",
    "11": "park",
    "12": "fitness center",
    "13": "clothes shop",
    "14": "beauty shop",
    "15": "water stream",
    "16": "river",
    "17": "bus stop",
    "18": "gas station",
    "19": "primary road",
    "20": "car parking place",
    "21": "public toilet"
}

def parse_gpt4(response):
    norm_response = {}
    for item in response:
        # check in the list
        lc_item = item.lower()
        for k, cands in POI_ALTER_NAME.items():
            if lc_item in cands:
                norm_response[POI_LIST[k]] = response[item]
    return norm_response

def parse_gemini(response):
    norm_response = {}
    for item in response:
        # check in the list
        lc_item = item.lower()
        for k, cands in POI_ALTER_NAME.items():
            if lc_item in cands:
                norm_response[POI_LIST[k]] = response[item]
    return norm_response 

def parse_internvl(response):
    norm_response = {}
    for item in response:
        # check in the list
        lc_item = item.lower()
        for k, cands in POI_ALTER_NAME.items():
            if lc_item in cands:
                norm_response[POI_LIST[k]] = response[item]
    return norm_response 