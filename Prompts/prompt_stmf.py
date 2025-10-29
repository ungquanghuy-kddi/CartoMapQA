POI_LIST_map = {
    "convenience stores": "convenience store",
    "supermarkets": "supermarket",
    "drug stores/pharmacies": "drug store/pharmacy",
    "general health clinics": "general clinic",
    "hospitals": "hospital",
    "restaurants": "restaurant",
    "fast food stores": "fast food store",
    "coffee shops": "coffee shop",
    "ATMs or cash points": "ATM or cash point",
    "banks": "bank",
    "playgrounds": "playground",
    "park areas": "park area",
    "fitness centers": "fitness center",
    "clothes shops": "clothes shop",
    "beauty shops": "beauty shop",
    "water streams": "water stream",
    "rivers": "river",
    "bus stops": "bus stop",
    "gas/fuel stations": "gas station",
    "primary roads": "primary road",
    "car parking places": "car parking lot",
    "public toilets": "public toilet"
}

TYPE = {
    "presence": 1,
    "counting": 2,
    "name_listing": 3,
}

def prompt_stmf(type, mf_type):
    qtype = TYPE[type]
    if qtype == 1: 
        prompt = f"""You are provided with a cartographic map sourced from OpenStreetMap.\n\
Your task is to determine whether the map contains any visual elements that indicate the presence of a point of interest (POI) of the following type: {mf_type}.\n\
Respond with "Yes" or "No" on the first line.\n\
On the next line, provide a concise explanation describing the evidence or absence of evidence that supports your answer.\n\
Base your reasoning on map features such as labels, icons, shapes, or patterns commonly associated with this type of POI."""

    if qtype == 2:
        prompt = f"""You are provided with a cartographic map sourced from OpenStreetMap.\n\
Your task is to identify and count all visual elements that indicate the presence of a point of interest (POI) of the following type: {mf_type}.\n\
On the first line of your reponse, provide the total number of such elements found.\n\
On the next line, explain your reasoning by describing the evidence used to support your count.\n\
Example of a response: \
5\n
[Explanation]"""

    if qtype == 3:
        prompt = f"""You are provided with a cartographic map sourced from OpenStreetMap. \n\
Your task is to identify all visual elements that indicate the presence of a point of interest (POI) of the following type: {mf_type}, and list those that have an associated name.\n\
Respond with one name per line, using the exact name as it appears on the map.\n\
If a visual element does not have a name, represent it with an empty string ("").\n\
If all relevant elements lack associated names, leave the output completely blank.\n\
Example format (when all names are present):\n\
Name_1\n\
Name_2\n\
...\n\
Example format (when some names are present):\n\
Name_1\n\
""\n\
Name_3\n\
...\n\
Example format (when no names are available):\n\
[leave the output blank]\n\
Strictly follow this format. Do not include any explanation, description, code, or additional text."""
    return prompt
