unit_abbreviations = {
    "meters": "m",
    "feet": "ft",
}
def prompt_rle(unit, use_CoT=True):
    if use_CoT:
        prompt = f"""You are provided with a cartographic map sourced from OpenStreetMap, where a blue path connects two blue map markers.\n\
Your task is to estimate the real-world length of this path in **{unit}** by following the steps below:\n\
1. Locate the map scale: Identify the scale notation (in {unit}) shown at the bottom-left corner of the map.\n\
2. Identify the path: Locate the two blue markers and the blue path that connects them.\n\
3. Measure the path: Measure the total length of this blue path on the map.\n\
4. Convert to real-world units: Apply the scale to convert your measured map-length to the actual distance in {unit}.\n\
Respond in the following format:\n\
- On the first line, provide the estimated length in the following format:\n\
[number] {unit_abbreviations[unit]}\n\
- On the second line, provide a brief explanation of the process you followed, referencing the use of the scale, \
measurement of the path, and the conversion applied.\n\
An output example can be: 625 {unit_abbreviations[unit]}\n\
The map scale indicates that 1 map unit equals 50 {unit_abbreviations[unit]}. \
I measured approximately 12.5 map units along the blue path, which converts to 625 {unit_abbreviations[unit]}."""
    else:
        prompt = f"""You are provided with a cartographic map sourced from OpenStreetMap. On the map, a blue path connects two blue map markers.\n\
Your task is to estimate the real-world length of this blue path and report it in **{unit}**.\n\
Respond with your estimated length followed by its units on the first line, using the following format:\n\
[number] {unit_abbreviations[unit]}\n\
On the second line, provide a concise explanation describing the methods you used to estimate this length."""
    return prompt