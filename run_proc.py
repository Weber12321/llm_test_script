import os
import pandas as pd
import datasets

def check_ans(string):
    check_chr = {'A', 'B', 'C', 'D'}
    return any(st not in check_chr for st in list(string))

def generate_column_names(length):
    if length < 1 or length > 26:
        raise ValueError("Length must be between 1 and 26")
    return ['question'] + [chr(i) for i in range(65, 65 + length)] + ['answer']

def transfer_to_df_dataset(
    data_path: str, 
    save_path: str, 
    single_choice: bool = True,
    only_four_choices: bool = False,
    split_test: bool = True,
    split_test_rate = 0.98
):
    df = pd.read_csv(data_path, encoding="utf-8").iloc[:, 1:]
    length = len(df.columns) - 2
    df.columns = generate_column_names(length)
    
    # filter choices
    if only_four_choices:
        df = df[df['answer'].apply(
            lambda x: False if check_ans(x) else True)
        ]
        df = df[['question', 'A', 'B', 'C', 'D', 'answer']]
    
    if single_choice:
        df = df[df['answer'].apply(
            lambda x: len(x) == 1
        )]

    dataset = datasets.Dataset.from_pandas(df)
    if split_test:
        dataset_split = dataset.train_test_split(test_size=split_test_rate)
        dev_dataset = dataset_split['train']
        test_dataset = dataset_split['test']

        dev_path = os.path.join(save_path, 'dev')
        test_path = os.path.join(save_path, 'test')
        os.makedirs(dev_path)
        os.makedirs(test_path)

        dev_dataset.save_to_disk(dev_path)
        test_dataset.save_to_disk(test_path)
    else:
        dataset.save_to_disk(save_path)


if __name__ == "__main__":
    data_path = "/home/tpiuser/expr/lm-evaluation-harness/datasets/csv/hr_data.csv"
    save_path = "/home/tpiuser/expr/lm-evaluation-harness/datasets/hf/hr_data"

    transfer_to_df_dataset(
        data_path, 
        save_path, 
        only_four_choices=False,
        split_test = True,
        split_test_rate = 0.99
    )
