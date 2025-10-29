import os
import json
import random

ind2ans = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E"
}

def evaluate_mfs(model_name="gemini-2.5-pro-preview-03-25", eval_dir="./Responses/MFS/gemini-2.5-pro-preview-03-25"):

    cr_count = 0
    n_question = 0
    with open(os.path.join(eval_dir, f"{model_name}.json"), 'r') as f:
        data = json.load(f)

    for item in data:
        question_info = data[item]
        if question_info['Answer'] == "":
            answer = ind2ans[random.randint(0, 4)]
        else:
            answer = question_info['Answer']

        if answer == question_info['Correct answer']:
            cr_count += 1

        n_question += 1

    acc = cr_count / n_question
    print ("============== MFS task ===============")
    print ("Model: {}".format(model_name))
    print ("Accuracy = {:0.3f} (correctly answer {} per total {} questions)".format(acc, cr_count, n_question))

    return

