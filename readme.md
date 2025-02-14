# CharacterBot

## Overview

We introduce CharacterBot, a model designed to replicate both the linguistic patterns and distinctive thought processes of a character. In this repository, we present Lu Xun—a renowned Chinese writer—as a case study.

## Dataset

The experimental dataset `luxun_essay.json` comprises 638 articles sourced from Wikisource. These collections span the entirety of Lu Xun’s mature essayistic output, reflecting a diverse array of themes from his intellectual career. All texts are in the public domain, ensuring unrestricted scholarly usage. As Lu Xun was a Chinese writer, the repository includes Chinese textual data and code annotations.

## Requirements

To set up the environment and install necessary dependencies, follow the steps below:

- `python3`
- `conda create --name env `
- `pip3 install -r requirements.txt`

## How to Use

### Data Generation

**Authorial Perspective Reframing**

Generate the Authorial Perspective Reframing pre-training data for the following commands:
```
cd characterbot
python authorial_perspective_reframing.py
```

**Multiple-choice Questions**

Generate data for the multiple-choice questions task:

```
python multiple_choice_questions_data.py
```

**Generative Question Answering** 

Generate data for the generative question answering task:

```
python generative_qa_data.py
```

**Style Transfer**

Generate data for the style transfer task:

```
python style_transfer_data.py
```


### Training

#### CharLoRA

CharLoRA requires modifications to three editable Python libraries. Make sure to download these libraries as specified in the project requirements. Then, apply the following updates:

##### Modifications for LLaMA-Factory:
```
cd /path/to/characterbot/train_with_charlora
cp aligner.py /path/to/LLaMA-Factory/src/llamafactory/data/aligner.py
cp collator.py /path/to/LLaMA-Factory/src/llamafactory/data/collator.py
cp supervised.py /path/to/LLaMA-Factory/src/llamafactory/data/processors/supervised.py
```

##### Modifications for PEFT:
```
cp peft_model.py /path/to/peft/src/peft/peft_model.py
cp save_and_load.py /path/to/peft/src/peft/utils/save_and_load.py
cp layer.py /path/to/peft/src/peft/tuners/lora/layer.py
```

##### Modifications for Transformers:
```
cp modeling_qwen2.py /path/to/transformers/src/transformers/models/qwen2/modeling_qwen2.py
```

#### Data Integration with LLaMA-Factory

Since both pre-training and fine-tuning processes rely on LLaMA-Factory, integrate your generated data into its structure.

Place data files into LLaMA-Factory's data directory:

```
cp pre_train.json /path/to/LLaMA-Factory/data/pre_train.json
cp fine_tune.json /path/to/LLaMA-Factory/data/fine_tune.json
cp test.json /path/to/LLaMA-Factory/data/test.json
cp dataset_info.json /path/to/LLaMA-Factory/data/dataset_info.json
```

Place training configuration files into the examples directory:

```
cp pre_train.yaml /path/to/LLaMA-Factory/examples/train_lora/pre_train.yaml
cp fine_tune.yaml /path/to/LLaMA-Factory/examples/train_lora/fine_tune.yaml
```

#### Pre-training

Run pre-training using the following command:

```
cd /path/to/LLaMA-Factory
llamafactory-cli train examples/train_lora/pre_train.yaml
```

#### Fine-tuning

Run fine-tuning with this command:

```
llamafactory-cli train examples/train_lora/fine-tuning.yaml
```

The output model will be saved at `/path/to/LLaMA-Factory/saves/fine_tune`


### Testing

#### Generating Responses

After training, you can generate responses for each task. Make sure to update the model and data paths as needed. Some data example format are shown in `example`. For each task, follow these instructions:

##### Multiple-choice Questions:

```
cd /path/to/characterbot/test/
cp peft_model_test.py /path/to/peft/src/peft/peft_model.py
cp mcq_load.py /path/to/peft/src/peft/utils/save_and_load.py
python multiple_choice_gen.py
```

##### Generative Question Answering:

```
cp gqa_load.py /path/to/peft/src/peft/utils/save_and_load.py
python generate_qa_gen.py
```

##### Style Transfer:

```
cp st_load.py /path/to/peft/src/peft/utils/save_and_load.py
python style_transfer_gen.py
```

#### Evaluation

Evaluate the performance of the model. Note that the evaluation code for multiple-choice questions is the same as the generation code. For the remaining tasks, use the following commands:

##### Generative Question Answering Evaluation:

```
python gen_qa_eval.py
```

##### Style Transfer Evaluation:

```
cd /path/to/characterbot/test/style_transfer_eval
python bleu.py
python rouge.py
python style_matching.py
```

## Acknowledgements
Parts of this project is built upon the following open-source libraries:
- [transformers](https://github.com/huggingface/transformers) (Apache 2.0)
- [peft](https://github.com/huggingface/peft) (Apache 2.0)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (Apache 2.0)