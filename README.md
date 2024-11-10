# Dynamic Rewarding with Prompt Optimization Enables Tuning-free Self-Alignment of Language Models

## Setup 

To setup the environment, you may use any of the following two ways (We use python 3.10):
1. `conda env create -f environment/conda_environment.yml`
2. `pip install -r environment/requirements.txt`

## API Keys/Access tokens

To use API based models/access restricted models, set-up the API Keys/access tokens as follows:
1. OpenAI API Key: `export OPENAI_API_KEY=<your key>`
2. Hugging Face Access Token: `export HF_TOKEN=<your token>`

## Accessing optimized prompts

To access the optimized prompts we have available, you may use:
```python
import pickle 

with open(<path_to_prompt_file>, 'rb') as f:
    prompt_obj =pickle.load(f)

try:
    model_prompt = prompt_obj.terminal_node.state[-1].system_prompt
except:
    model_prompt = prompt_obj
```

## Accessing optimized ICL examples

ICL examples can be accessed at `data/ICL_examples.json`