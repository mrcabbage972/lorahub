import os

from lorahub.algorithm import lorahub_learning, lorahub_inference
import random
import datasets
import multiprocessing
import json

def get_examples_for_learning(ood_dataset_path, target_task_name, num_train_examples=5, num_inference_examples=100):
    """
    Get a few examples to learn to compose given LoRA modules
    """

    ds = datasets.load_from_disk(ood_dataset_path)
    ds = ds.filter(lambda x: x['task_name'] == target_task_name, num_proc=multiprocessing.cpu_count())

    assert len(ds) > 0, 'Task not found in dataset'

    ds_train = ds.select(range(num_train_examples))
    ds_test = ds.select(range(num_train_examples, num_train_examples + num_inference_examples))

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

    results = []

    for target_task_name in target_task_names:
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
    
    with open("lorahub_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    loras_path = '/home/ubuntu/victor/retrieval-of-experts/data/loras'
    ood_dataset_path = '/home/ubuntu/victor/retrieval-of-experts/data/flanv2_tokenized_new_splits/ood'
    target_task_names = ['aeslc:1.0.0', 'wmt14_translate/fr-en:1.0.0',
       'duorc_ParaphraseRC_generate_question',
       'quail_context_question_description_text', 'openbookqa:0.1.0',
       'cos_e_v1.11_aligned_with_common_sense',
       'opinion_abstracts_idebate', 'wiki_qa_automatic_system',
       'race_high_Write_a_multi_choice_question_for_the_following_article',
       'gem/dart:1.1.0',
       'race_high_Read_the_article_and_answer_the_question_no_option_',
       'super_glue/record:1.0.2', 'duorc_ParaphraseRC_answer_question',
       'quail_no_prompt_id', 'quail_description_context_question_text',
       'quartz_answer_question_below', 'quoref_Guess_Answer',
       'cos_e_v1.11_rationale',
       'cos_e_v1.11_question_description_option_id',
       'ai2_arc/ARC-Challenge:1.0.0', 'amazon_polarity_user_satisfied',
       'glue/wnli:2.0.0', 'wiki_hop_original_explain_relation',
       'wiki_qa_Topic_Prediction_Answer_Only', 'anli/r2:0.1.0']
    main(loras_path, ood_dataset_path, target_task_names)
