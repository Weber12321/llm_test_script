
from config import task_list
from datasets import load_dataset




for task in task_list:
    val = load_dataset('ikala/tmmluplus', task)['validation']
    dev = load_dataset('ikala/tmmluplus', task)['train']
    test = load_dataset('ikala/tmmluplus', task)['test']