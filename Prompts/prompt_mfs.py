
def prompt_mfs(A, B, C, D, E, question):
    prompt = f"""Answer the following multiple-choice question: {question} \n\
A. {A} \n\
B. {B} \n\
C. {C} \n\
D. {D} \n\
E. {E} \n\
Respond by a single letter: A or B or C or D or E.\n\
For example: \n\
A"""
    return prompt