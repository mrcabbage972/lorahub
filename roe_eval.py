import os

from lorahub.algorithm import lorahub_learning, lorahub_inference
import random
import datasets
import multiprocessing
import json
import torch
import gc
import pickle

def get_examples_for_learning(ds, target_task_name, num_train_examples=5, num_inference_examples=100):
    """
    Get a few examples to learn to compose given LoRA modules
    """

    #ds = datasets.load_from_disk(ood_dataset_path)
    ds = ds.filter(lambda x: x['task_name'] == target_task_name, num_proc=multiprocessing.cpu_count())

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
    
def get_lora_module_list_retrieved(ood_dataset_eval, text_ds, target_task_names, train_examples):
    
    # Get the indices of train_examples
    text_inputs = text_ds['inputs']
    train_example_indices = [text_inputs.index(example['input']) for example in train_examples]

    # Get the relevant rows in the retrieved ranks
    retrieved_ranks = ood_dataset_eval['named_ranks']
    relevant_rows = [retrieved_ranks[i] for i in train_example_indices]

    # For each module, get all the ranks at which it was retrieved
    module_ranks = {}
    for row in relevant_rows:
        for i, module in enumerate(row):
            if module in target_task_names:
                continue # Skipping OOD tasks
            if module not in module_ranks:
                module_ranks[module] = []
            module_ranks[module].append(i)

    # Calculate the mean rank for each module and return the top 20
    module_mean_ranks = {module: sum(ranks) / len(ranks) for module, ranks in module_ranks.items()}
    sorted_modules = sorted(module_mean_ranks.items(), key=lambda x: x[1])
    return [module for module, _ in sorted_modules[:20]]

    



def main(task_lora_path, ood_dataset_path, target_task_name, ood_dataset_eval=None):
    """
    Perform lorahub learning
    """
    #task_text_path = '/home/ubuntu/victor/retrieval-of-experts/data/split_loras'
    text_ds = datasets.load_from_disk(ood_dataset_path)
    
    results = []

    for target_task_name in target_task_names:
        train_examples, inference_examples = get_examples_for_learning(text_ds, target_task_name)

        # get a list of modules to be used in the composition
        if ood_dataset_eval is None:
            print('Using fixed module list')
            modules = get_lora_module_list_fixed(loras_path)
        else:
            print('Using retrieved module list')
            modules = get_lora_module_list_retrieved(ood_dataset_eval, text_ds, target_task_names, train_examples)
        print("modules:", modules)

        gc.collect()
        torch.cuda.empty_cache()
        

        # construct input list and output list
        example_inputs, examples_outputs = [], []
        for example in train_examples:
            example_inputs.append(example["input"])
            examples_outputs.append(example["output"])

        # perform LoRAHub learning
        module_weights, model, tokenizer = lorahub_learning(task_lora_path=task_lora_path,
                                                            lora_module_list=modules,
                                                            example_inputs=example_inputs,
                                                            example_outputs=examples_outputs,
                                                            max_inference_step=40,
                                                            batch_size=5)

        #print("module_weights:", module_weights)

        """
        Perform inference to get predictions
        """
        # now you can use the model to perform inference
        example_inputs, examples_outputs = [], []
        for example in inference_examples:
            example_inputs.append(example["input"])
            examples_outputs.append(example["output"])

        example_predictions, perf = lorahub_inference(example_inputs=example_inputs,
                                                    model_or_name_path=model,
                                                    tokenizer_or_tokenizer_path=tokenizer,
                                                    batch_size=10,
                                                    # can set as None if you do not have the ground truth
                                                    example_outputs=examples_outputs)
        print("task_name:", target_task_name)
        print("example_predictions:", example_predictions)
        print("task accuracy:", perf)

        results.append({"task_name": target_task_name, "task_accuracy": perf})
    
        filename_suffix = "retrieved" if ood_dataset_eval is not None else "fixed"
        filename = f"lorahub_results_{filename_suffix}.json"
        with open(filename, 'w') as f:
            json.dump(results, f)


if __name__ == "__main__":
    loras_path = '/home/ubuntu/victor/retrieval-of-experts/data/loras'
    #ood_dataset_path = '/home/ubuntu/victor/retrieval-of-experts/data/flanv2_tokenized_new_splits/ood'
    ood_dataset_path = '/home/ubuntu/victor/retrieval-of-experts/data/split_loras/ood'
    target_task_names = ['duorc_ParaphraseRC_generate_question',
       'quail_context_question_description_text',
       'race_high_Write_a_multi_choice_question_for_the_following_article',
       'race_high_Read_the_article_and_answer_the_question_no_option_',
       'duorc_ParaphraseRC_answer_question', 'quail_no_prompt_id',
       'quail_description_context_question_text',
       'quartz_answer_question_below', 'quoref_Guess_Answer',
       'amazon_polarity_user_satisfied']
    

    ood_dataset_eval_path = '/home/ubuntu/victor/eval_output_dict.p'
    with open(ood_dataset_eval_path, "rb") as f:
        ood_dataset_eval = pickle.load(f)

    #main(loras_path, ood_dataset_path, target_task_names, ood_dataset_eval)
    main(loras_path, ood_dataset_path, target_task_names)
