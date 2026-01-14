# Taken and adapted from  https://github.com/xzymustbexzy/Chain-of-Experts.git : @inproceedings{
# xiao2024chainofexperts,
# title={Chain-of-Experts: When {LLM}s Meet Complex Operations Research Problems},
# author={Ziyang Xiao and Dongxiang Zhang and Yangjun Wu and Lilin Xu and Yuan Jessica Wang and Xiongwei Han and Xiaojin Fu and Tao Zhong and Jia Zeng and Mingli Song and Gang Chen},
# booktitle={The Twelfth International Conference on Learning Representations},
# year={2024},
# url={https://openreview.net/forum?id=HobyL1B9CZ}
# }


from enum import Enum


class Result(Enum):

    ACCEPT = 0
    WRONG_ANSWER = 1
    RUNTIME_ERROR = 2
    COMPILE_ERROR = 3
