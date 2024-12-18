# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import pickle

import torch.distributed as dist

from align_anything.evaluation.data_type import InferenceInput
from align_anything.evaluation.dataloader.base_dataloader import BaseDataLoader
from align_anything.evaluation.inference.ds_inference import BaseInferencer_deepspeed
from align_anything.utils.template_registry import get_eval_template_class as get_template_class
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    read_eval_cfgs,
    update_dict,
)
from datasets import DatasetDict, load_dataset


class ARCDataLoader(BaseDataLoader):
    def get_task_names(self):
        if isinstance(self.data_cfgs.task, list):
            return self.data_cfgs.task
        else:
            task_names = [self.data_cfgs.task]
            return task_names

    def get_answer(self, data):
        return data['answerKey']

    def set_fewshot_dataset(self, dataset, task):
        return dataset['validation']

    def build_example_prompt(self, data, with_answer=True, cot=False):
        choices = get_choices(data)
        answer = f'Answer: {self.get_answer(data)}' if with_answer else 'Answer: '
        return f"{data['question']}Please choose the correct answer from the following options:\n{choices}\n{answer}"

    def build_prompt(self, data):
        prompt = ''
        cot_prompt = f" Let's think step by step. "
        few_shot_examples = self.few_shot_data[: self.num_shot] if self.num_shot else []
        template = get_template_class(self.chat_template)
        if len(few_shot_examples) == 0:
            question = [
                template.system_prompt
                + template.user_prompt.format(input=prompt + self.build_example_prompt(item, False))
                + template.assistant_prompt.format(output='')
                for item in data
            ]
        else:
            if not self.cot:
                few_shots = [
                    self.build_example_prompt(
                        {key: value[i] for key, value in few_shot_examples.items()}, True
                    )
                    for i in range(len(few_shot_examples['question']))
                ]
            else:
                few_shots = [
                    f"{example['question']}\n'Answer: '{example['answer']}"
                    for example in few_shot_examples
                ]
            question = []
            for item in data:
                request = {}
                for key, value in item.items():
                    request[key] = value
                examples = few_shots + [self.build_example_prompt(request, False)]
                if self.cot:
                    question.append(
                        template.system_prompt
                        + template.user_prompt.format(input=prompt + '\n\n'.join(examples))
                        + template.assistant_prompt.format(output=cot_prompt)
                    )
                else:
                    question.append(
                        template.system_prompt
                        + template.user_prompt.format(input=prompt + '\n\n'.join(examples))
                        + template.assistant_prompt.format(output='')
                    )

        return question

    def load_dataset(self, eval_module) -> DatasetDict:
        for task in self.task_names:
            dataset = load_dataset(self.task_dir, task)
            self.few_shot_data = self.set_fewshot_dataset(dataset, task)
            prompts, token_ids = self.preprocess(dataset)
            processed_inputs = [
                InferenceInput(text=prompt, token_ids=token_id)
                for prompt, token_id in zip(prompts, token_ids['input_ids'])
            ]
            eval_module.save_data(task, processed_inputs)


class ARCGeneratorDS(BaseInferencer_deepspeed):
    def save_data(self, task, data):
        os.makedirs('.cache', exist_ok=True)

        task_dir = f'.cache/{task}'
        os.makedirs(task_dir, exist_ok=True)
        InferenceOutputs = self.generation(data)
        if dist.is_initialized():
            file_path = f'{task_dir}/outputs_{get_rank()}.pkl'
        else:
            file_path = f'{task_dir}/outputs.pkl'

        with open(file_path, 'wb') as f:
            pickle.dump(InferenceOutputs, f, protocol=4)


def get_choices(data):
    if data['choices']['label'][0] == 'A':
        choices = '\n' + '\n'.join(
            [
                f"({chr(label+65)}) {data['choices']['text'][label]}"
                for label in range(len(data['choices']['text']))
            ]
        )
    else:
        choices = '\n' + '\n'.join(
            [
                f"({label}) {data['choices']['text'][label]}"
                for label in range(len(data['choices']['text']))
            ]
        )
    return choices


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[1::2]]
    values = list(unparsed_args[2::2])
    unparsed_args = dict(zip(keys, values))
    dict_configs, infer_configs = read_eval_cfgs('arc', 'deepspeed')
    for k, v in unparsed_args.items():
        if v == '' or v is None:
            continue
        dict_configs = update_dict(dict_configs, custom_cfgs_to_dict(k, v))
        infer_configs = update_dict(infer_configs, custom_cfgs_to_dict(k, v))

    dict_configs = dict_to_namedtuple(dict_configs)
    model_config = dict_configs.default.model_cfgs
    dataloader = ARCDataLoader(dict_configs)
    assert not dataloader.cot, 'chain-of-thought cannot be used for this benchmark.'
    eval_module = ARCGeneratorDS(model_config, infer_configs)
    dataloader.load_dataset(eval_module)


if __name__ == '__main__':
    main()
