import os
from typing import List
import subprocess
import wandb

def run_lm_eval(
    run_name: str, 
    expr_name: str,
    model_path: str, 
    tasks: List[str], 
    device: str = "cuda:0", 
    batch_size: int = 4,
    output_path: str = 'results/hf',
    **kwargs
) -> None:
    
    if not os.path.isdir(model_path):
        print(f"{model_path} is not found in current directory, skip the task...")
        return
        # raise NotADirectoryError(
        #     "model_path %s not found!", model_path
        # )

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    max_new_tokens = kwargs.get("max_new_tokens", 200)
    top_p = kwargs.get("top_p", 0.001)
    top_k = kwargs.get("top_k", 0)
    temperature = kwargs.get("temperature", 0.9)


    model_args = ",".join([
        f"pretrained={model_path}",
    ])

    gen_args = ",".join([
        f"temperature={temperature}",
        f"top_k={top_k}",
        f"top_p={top_p}",
        f"max_new_tokens={max_new_tokens}"
    ])

    command = [
        'lm_eval',
        '--model', 'hf',
        '--model_args',  model_args,
        '--gen_kwargs', gen_args,
        '--tasks', ','.join(tasks),
        '--device', device,
        '--batch_size', str(batch_size),
        '--output_path', output_path,
        '--limit', '10',
        '--wandb_args', f'project=lm-eval-harness-{expr_name},name={run_name.lower()}',
        '--log_samples'
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)


if __name__ == "__main__":
    wandb.login()
    
    # expr_name = 'tmlu'
    expr_name = 'hr_spec'

    # task_list = ['tmmluplus',]
    task_list = ['hr_spec',]
    # task_list = ['tmlu']

    models = {
        'llama3': './models/Meta-Llama-3-8B-Instruct',    
        'taiwanllm': './models/Taiwan-LLM-7B-v2.1-chat',
        'breeze': './models/Breeze-7B-Instruct-v0_1',
        'taide': './models/TAIDE-LX-7B-Chat'
    }

    for run_name, model_path in models.items():

        run_lm_eval(
            run_name=run_name,
            expr_name=expr_name,
            model_path=model_path,
            tasks=task_list
        )


