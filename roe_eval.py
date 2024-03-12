import os

from lorahub.algorithm import lorahub_learning, lorahub_inference
import random
import datasets


def get_examples_for_learning(ood_dataset_path, target_task_name, num_train_examples=5):
    """
    Get a few examples to learn to compose given LoRA modules
    """

    ds = datasets.load_from_disk(ood_dataset_path)
    ds = ds.filter(lambda x: x['task_name'] == target_task_name)

    assert len(ds) > 0, 'Task not found in dataset'

    ds_train = ds.select(range(num_train_examples))
    ds_test = ds.select(range(num_train_examples, len(ds)))

    train_examples = [{'input': input_, 'output': output} for input_, output in
                      zip(ds_train['inputs'], ds_train['targets'])]

    inference_examples = [{'input': input_, 'output': output} for input_, output in
                          zip(ds_test['inputs'], ds_test['targets'])]

    return train_examples, inference_examples


def get_lora_module_list(task_lora_path):
    """
    You can have a custom filtering strategy to select the modules to be used in the composition. Here we randomly select 20 modules.
    """

    lora_module_names = [x for x in os.listdir(task_lora_path) if os.path.isdir(os.path.join(task_lora_path, x))]
    assert len(lora_module_names) > 0, f'No tasks found in {task_lora_path}'

    random.seed(42)
    return random.sample(lora_module_names, 20)


def main(task_lora_path, ood_dataset_path, target_task_name):
    """
    Perform lorahub learning
    """
    # get a list of modules to be used in the composition
    modules = get_lora_module_list(loras_path)
    print("modules:", modules)

    train_examples, inference_examples = get_examples_for_learning(ood_dataset_path, target_task_name)

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

    print("module_weights:", module_weights)

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
    print("example_predictions:", example_predictions)
    print("task accuracy:", perf)


if __name__ == "__main__":
    loras_path = '/Users/vmay/Documents/git/retrieval-of-experts/data/loras'
    ood_dataset_path = '/Users/vmay/Documents/git/retrieval-of-experts/data/flanv2_tokenized/validation'
    target_task_name = 'social_i_qa_Check_if_a_random_answer_is_valid_or_not'
    main(loras_path, ood_dataset_path, target_task_name)
