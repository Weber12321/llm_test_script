import sys
sys.path.append('/expr/apps')

from utils.evaluate import HF_Evaluator
from ievals.exp_executer import run_exp


def main():
    dataset="ikala/tmmluplus"
    model_name = "/expr/apps/storage/models/yentinglin/Taiwan-LLM-7B-v2.1-chat"
    top_k = 0
    cot = False  # chain of thought
    switch_zh_hans = False
    cache=True

    evaluator = HF_Evaluator(
        choices="A,B,C,D",
        k=0,
        api_key=None,
        model_name=model_name,
        switch_zh_hans=switch_zh_hans,
    )

    postfix = model_name.split("/")[-1]
    if top_k > 0:
        postfix += f"_top_{top_k}"
    if cot:
        postfix += "_cot"

    cache_path = None
    if cache:
        cache_path = "/expr/apps/.cache"
        if top_k > 0:
            cache_path += f"_top_{top_k}"
        if cot:
            cache_path += "_cot"
        if switch_zh_hans:
            cache_path += "_zhs"
    
    run_exp(
        evaluator,
        model_name,
        dataset,
        cot=cot,
        few_shot=top_k > 0,
        cache_path=cache_path,
        postfix_name=postfix,
        switch_zh_hans=switch_zh_hans,
    )


if __name__ == "__main__":
    main()




