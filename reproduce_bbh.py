from lorahub.algorithm import lorahub_inference
import os
import json
from lorahub.algorithm import lorahub_learning, lorahub_inference
from lorahub.constant import LORA_MODULE_NAMES
import random
import datasets
import pickle


def evaluate_flan_results_zero_shot(folder, flan_model_name):
    sub_dirs = os.listdir(folder)

    for sub_dir in sub_dirs:
        test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
        print("Evaluating on task (zero shot): ", sub_dir)
        lorahub_inference(task_inputs,
                          flan_model_name,
                          flan_model_name,
                          16,
                          task_outputs)


def evaluate_flan_results_few_shot(folder, flan_model_name):
    sub_dirs = os.listdir(folder)

    for sub_dir in sub_dirs:
        test_file_path = os.path.join(folder, sub_dir, "few_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
        print("Evaluating on task (five shot): ", sub_dir)
        lorahub_inference(task_inputs,
                          flan_model_name,
                          flan_model_name,
                          16,
                          task_outputs)


def get_lora_module_list(loras_path, folder, sub_dirs, example_inputs, retrieval_eval):
    if isinstance(loras_path, str):
        lora_module_names = [x for x in os.listdir(loras_path) if os.path.isdir(os.path.join(loras_path, x))]
        assert len(lora_module_names) > 0, f'No tasks found in {loras_path}'
        if retrieval_eval:
            return get_lora_module_list_retrieved(retrieval_eval, rank_dataset, folder, sub_dirs, example_inputs)
        else:
            return random.sample(lora_module_names, 20)
    else:
        return random.sample(loras_path, 20)


def get_lora_module_list_retrieved(retrieval_eval, rank_dataset, folder, target_task_names, example_inputs):

    # Get the indices of train_examples
    text_ds = datasets.load_from_disk(rank_dataset)
    text_inputs = text_ds['inputs']
    train_example_indices = [text_inputs.index(example) for example in example_inputs]

    # Get the relevant rows in the retrieved ranks
    with open(retrieval_eval, "rb") as f:
        retrieval_eval_data = pickle.load(f)
    retrieved_ranks = retrieval_eval_data['named_ranks']
    relevant_rows = [retrieved_ranks[i] for i in train_example_indices]

    # For each module, get all the ranks at which it was retrieved
    module_ranks = {}
    for row in relevant_rows:
        for i, module in enumerate(row):
            if module in target_task_names:
                continue  # Skipping OOD tasks
            if module not in module_ranks:
                module_ranks[module] = []
            module_ranks[module].append(i)

    # Calculate the mean rank for each module and return the top 20
    module_mean_ranks = {module: sum(ranks) / len(ranks) for module, ranks in module_ranks.items()}
    sorted_modules = sorted(module_mean_ranks.items(), key=lambda x: x[1])
    print("Sorted Modules:", len(sorted_modules))
    print(sorted_modules[:20])
    return [module for module, _ in sorted_modules[:20]]


def prepare_data(folder, file_path):
    task_inputs, task_outputs = [], []
    for line in open(file_path, "r", encoding="utf-8"):
        example = json.loads(line)
        task_inputs.append(example["context"])
        task_outputs.append(example["completion"])

    print(f"Dataset for {file_path} loaded of size:", len(task_inputs), len(task_outputs))
    return task_inputs, task_outputs


def evaluate_lorahub_results_few_shot(folder, loras_path=LORA_MODULE_NAMES, retrieval_eval=None, rank_dataset=None):

    results = []

    sub_dirs = os.listdir(folder)

    # 5 seeds used in our experiments
    for sub_dir in sub_dirs:
        # construct the few-shot examples for lorahub learning
        print("Evaluating on task (few shot): ", sub_dir)
        file_path = os.path.join(folder, sub_dir, "example.jsonl")
        example_inputs, examples_outputs = prepare_data(folder, file_path)

        # random select 5 examples for each task
        random.seed(42)
        shuffled_set = list(zip(example_inputs, examples_outputs))
        random.shuffle(shuffled_set)
        example_inputs, examples_outputs = zip(*shuffled_set)
        # take the first 5 examples
        example_inputs, examples_outputs = example_inputs[:5], examples_outputs[:5]

        # load the zero-shot examples for evaluation
        file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs = prepare_data(folder, file_path)

        task_perf_list = []
        for seed in range(1, 6):
            random.seed(seed)

            # get a list of modules to be used in the composition
            modules = get_lora_module_list(loras_path, folder, sub_dirs, example_inputs, retrieval_eval)

            # perform LoRAHub learning
            module_weights, model, tokenizer = lorahub_learning(task_lora_path=loras_path,
                                                                lora_module_list=modules,
                                                                example_inputs=example_inputs,
                                                                example_outputs=examples_outputs,
                                                                max_inference_step=40,
                                                                batch_size=5)

            print("module_weights:", module_weights)

            """
            Perform inference to get predictions
            """
            _, task_acc = lorahub_inference(example_inputs=task_inputs,
                                            model_or_name_path=model,
                                            tokenizer_or_tokenizer_path=tokenizer,
                                            batch_size=10,
                                            # can set as None if you do not have the ground truth
                                            example_outputs=task_outputs)
            task_perf_list.append(task_acc)
            
            if retrieval_eval:
                break
        avg_perf, max_perf = sum(task_perf_list) / len(task_perf_list), max(task_perf_list)
        print("Evaluating on task (few shot): ", sub_dir)
        print("average perf:", avg_perf, "best perf:", max_perf)
        results.append({"task_name": sub_dir, "average_perf": avg_perf, "best_perf": max_perf})

        suffix = "_custom" if isinstance(loras_path, str) else ""
        ret_suffix = "_ret" if retrieval_eval else ""
        filename = f"lorahub_results_bbh{suffix}{ret_suffix}.json"
        with open(filename, 'w') as f:
            json.dump(results, f)


if __name__ == "__main__":
    # evaluate the model
    # evaluate_flan_results_zero_shot("data_bbh", "google/flan-t5-large")

    # five shot for flan models
    # evaluate_flan_results_few_shot("data_bbh", "google/flan-t5-large")

    # five shot for lorahub models
    
    data_base_path = "/home/ubuntu/efs/anshita/retrieval-of-experts/data"

    #new loras path
    loras_path = os.path.join(data_base_path, 'loras')

    #old LORAHUB LoRA
    # loras_path = LORA_MODULE_NAMES

    dataset = os.path.join(data_base_path, 'data_bbh')

    retrieval_eval = os.path.join(data_base_path, 'lightning_checkpoints/rose-elevator-105/epoch=1-step=8712_retrieval_output_dict_data_bbh_tokenized.p')
    
    rank_dataset = os.path.join(data_base_path, 'data_bbh_tokenized')

    #retrieval result
    evaluate_lorahub_results_few_shot(dataset, loras_path, retrieval_eval, rank_dataset)

    #without retrieval result
    evaluate_lorahub_results_few_shot(dataset, loras_path)
