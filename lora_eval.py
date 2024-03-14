import os

from lorahub.algorithm import lorahub_learning, lorahub_inference
import random
import datasets
import multiprocessing
import json
import torch
import gc
import os

from transformers import AutoModelForSeq2SeqLM
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy
import random
import nevergrad as ng
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from peft import PeftModel, PeftConfig
from functools import partial
from typing import List, Optional, Union
import copy
import pickle

def get_examples_for_learning(ds, target_task_name, num_train_examples=5, num_inference_examples=100):
    """
    Get a few examples to learn to compose given LoRA modules
    """

    #ds = datasets.load_from_disk(ood_dataset_path)
    ds = ds.filter(lambda x: x['clean_task_name'] == target_task_name, num_proc=multiprocessing.cpu_count())

    assert len(ds) > 0, 'Task not found in dataset'

    ds_train = ds.select(range(num_train_examples))
    ds_test = ds.select(range(num_train_examples, num_train_examples + num_inference_examples))

    train_examples = [{'input': input_, 'output': output} for input_, output in
                      zip(ds_train['inputs'], ds_train['targets'])]

    inference_examples = [{'input': input_, 'output': output} for input_, output in
                          zip(ds_test['inputs'], ds_test['targets'])]

    return train_examples, inference_examples


def get_lora_module_list_fixed(task_lora_path):
    """
    You can have a custom filtering strategy to select the modules to be used in the composition. Here we randomly select 20 modules.
    """
    
    random.seed(42)
    
    lora_module_names = [x for x in os.listdir(task_lora_path) if os.path.isdir(os.path.join(task_lora_path, x))]
    assert len(lora_module_names) > 0, f'No tasks found in {task_lora_path}'
    return random.sample(lora_module_names, 20)
    

def main(task_lora_path, ood_dataset_path, ood_dataset_eval=None):
    """
    Perform lorahub learning
    """
    #task_text_path = '/home/ubuntu/victor/retrieval-of-experts/data/split_loras'
    text_ds = datasets.load_from_disk(ood_dataset_path)

    target_task_names = text_ds.to_pandas()['clean_task_name'].unique()

    print("target_task_names")
    print(target_task_names)
    
    results = []

    for target_task_name in target_task_names:
        print("target_task_name:", target_task_name)

        train_examples, inference_examples = get_examples_for_learning(text_ds, target_task_name)

       
        modules = get_lora_module_list_fixed(loras_path)


        gc.collect()
        torch.cuda.empty_cache()
        

        # construct input list and output list
        example_inputs, examples_outputs = [], []
        for example in train_examples:
            example_inputs.append(example["input"])
            examples_outputs.append(example["output"])

        # perform LoRAHub learning
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # load basic model
        default_peft_model_id = os.path.join(task_lora_path, target_task_name)
        # find the base model
        
        model_name_or_path = PeftConfig.from_pretrained(default_peft_model_id).base_model_name_or_path
            
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # 0 is the default model
        try:
            peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id)
        except:
            raise Exception(f'{default_peft_model_id} is unable to load into the model {model_name_or_path}')
            
        peft_model = peft_model.to(device)
        peft_model.eval()


        """
        Perform inference to get predictions
        """
        # now you can use the model to perform inference
        example_inputs, examples_outputs = [], []
        for example in inference_examples:
            example_inputs.append(example["input"])
            examples_outputs.append(example["output"])

        example_predictions, perf = lorahub_inference(example_inputs=example_inputs,
                                                    model_or_name_path=peft_model,
                                                    tokenizer_or_tokenizer_path=tokenizer,
                                                    batch_size=10,
                                                    # can set as None if you do not have the ground truth
                                                    example_outputs=examples_outputs)
        print("task_name:", target_task_name)
        print("example_predictions:", example_predictions)
        print("task accuracy:", perf)

        results.append({"task_name": target_task_name, "task_accuracy": perf})
    
        filename_suffix = "retrieved" if ood_dataset_eval is not None else "fixed"
        filename = f"lorah_results_{filename_suffix}.json"
        with open(filename, 'w') as f:
            json.dump(results, f)


if __name__ == "__main__":
    loras_path = '/home/ubuntu/victor/retrieval-of-experts/data/loras'
    ood_dataset_path = '/home/ubuntu/victor/retrieval-of-experts/data/flanv2_tokenized_new_splits/ood'
   

    ood_dataset_eval_path = '/home/ubuntu/victor/retrieval-of-experts/eval_output_dict_test.p'
    with open(ood_dataset_eval_path, "rb") as f:
        ood_dataset_eval = pickle.load(f)

    main(loras_path, ood_dataset_path, ood_dataset_eval)
    #main(loras_path, ood_dataset_path)
