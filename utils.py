# Taken from  https://github.com/xzymustbexzy/Chain-of-Experts.git : @inproceedings{
# xiao2024chainofexperts,
# title={Chain-of-Experts: When {LLM}s Meet Complex Operations Research Problems},
# author={Ziyang Xiao and Dongxiang Zhang and Yangjun Wu and Lilin Xu and Yuan Jessica Wang and Xiongwei Han and Xiaojin Fu and Tao Zhong and Jia Zeng and Mingli Song and Gang Chen},
# booktitle={The Twelfth International Conference on Learning Representations},
# year={2024},
# url={https://openreview.net/forum?id=HobyL1B9CZ}
# }


import re
import json
import os


def extract_code_from_string(input_string):
    # Match code within ```python ... ``` or ``` ... ``` blocks
    pattern = r'```(?:python)?\s*(.*?)\s*```'
    
    # Find all matches in the input string
    code_blocks = re.findall(pattern, input_string, re.DOTALL)

    if len(code_blocks) == 0:
        # print(f'Parse code error! {input_string}')
        return input_string
    elif len(code_blocks) == 1:
        return code_blocks[0]

    code_blocks = [code for code in code_blocks if 'pip' not in code]
    return '\n'.join(code_blocks)


def read_problem(dataset, problem_name):
    base_dir = 'dataset'
    with open(os.path.join(base_dir, dataset, problem_name, 'description.txt'), 'r', encoding='utf8') as f:
        description = f.read()

    with open(os.path.join(base_dir, dataset, problem_name, 'code_example.py'), 'r', encoding='utf8') as f:
        code_example = f.read()

    return {
        'description': description,
        'code_example': code_example
    }
