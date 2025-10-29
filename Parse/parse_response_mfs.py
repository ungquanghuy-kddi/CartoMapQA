
def parse_mfs(response):
    answer = response.strip()[0]
    if answer not in ['A', 'B', 'C', 'D', 'E']:
        if "correct answer is" in response:
            answer = response.strip().split("correct answer is ")[1].strip().replace(".", "")
            if answer not in ['A', 'B', 'C', 'D', 'E']:
                answer = ""
        else:
            answer = ""
    
    return answer