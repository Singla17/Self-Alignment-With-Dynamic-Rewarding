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

To access the optimized prompts we have available, you may use the following code snippet:
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


## Inference with the optimized prompt
We show an example of how to use the `AutoModel` API for inference:

```python
from auto_model import AutoModel

model = AutoModel( model_name = "mistralai/Mistral-7B-v0.1",
        num_gpus = 1,
        cuda_visible_devices = "0",
        dtype = 'bfloat16',
        gpu_memory_utilization = 0.5,
)

print(model.generate(
        user_query = "Plan a 7-day trip in India.",
        optimized_prompt = True,
        optimized_icl = True,
        num_optimized_icl = 3,
        temperature = 0.7,
        top_p = 0.95,
        max_new_tokens = 512,
))
```

If you want to use a custom system prompt: 

```python
from auto_model import AutoModel

model = AutoModel( model_name = "mistralai/Mistral-7B-v0.1",
        num_gpus = 1,
        cuda_visible_devices = "0",
        dtype = 'bfloat16',
        gpu_memory_utilization = 0.5,
)

print(model.generate(
        user_query = "Plan a 7-day trip in India.",
        user_specified_system_prompt = "You are a helpful assistant",
        optimized_icl = True,
        num_optimized_icl = 3,
        temperature = 0.7,
        top_p = 0.95,
        max_new_tokens = 512,
))
```


If you want to use the API for an Open-AI model: 

```python
from auto_model import AutoModel

model = AutoModel( model_name = "gpt-3.5-turbo-0125",
        timeout = 600,
        open_ai_model = True,
)

print(model.generate(
        user_query= "Plan a 7-day trip in India.",
        optimized_prompt = True,
        optimized_icl = True,
        num_optimized_icl = 3,
        temperature = 0.7,
        top_p = 0.95,
        max_new_tokens = 512,
))
```
