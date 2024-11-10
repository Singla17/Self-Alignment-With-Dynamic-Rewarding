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

Note: For this code to be usable the `reasoners` folder should be in the working directory.

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
Note: All the scripts have been tested on a single A100 GPU with 80GB memory. If the scripts fail on your GPU, it might be worth playing with `num_gpus` and `gpu_memory_utilization` parameters, these are as defined in the [vLLM API](https://github.com/vllm-project/vllm). 

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

## Training the optimized prompt for a model

To optimize the alignment prompt for `Mistral-7B` you may use the shell script and you can update the script as per your needs:

`bash prompt_train.sh`

All the parameters are explain as follows:

```
- base_model_name (str): Name or path of the base model to be used.
- base_model_family (str): Family name of the base model, e.g., 'mistral'.
- eval_model_name (str): Model name for evaluation purposes, e.g., 'gpt-4-0125-preview'.
- metrics_model_name (str): Model name for dynamic reward selection.
- optimize_model_name (str): Model name used for optimization tasks.
- initial_system_prompt (str): Initial system prompt for the model.
- n_actions (int): Number of actions to be sampled in the beam search.
- temperature (float): Temperature for controlling randomness in model predictions.
- depth (int): Initial search depth for exploration.
- max_depth_increase (int): Maximum increment allowed in search depth. (Used when the original training samples are of low difficulty)
- beam_size (int): Number of beams for beam search.
- log_dir (Optional[str]): Directory path for storing logs.
- disable_log (bool): If True, disables logging.
- disable_tqdm (bool): If True, disables tqdm progress bars.
- base_model_download_dir (str): Directory for downloading base model files.
- data_dir (str): Directory path for data files.
- metrics_cache_path (str): File path to cache evaluation metrics.
- num_training_examples (int): Number of examples to use for training.
- logging_level (str): Logging level, e.g., "INFO" or "DEBUG".
- ret_icl (bool): If True, then prompt is optimized with retreival based ICL.
- is_GPT (bool): If True, treats the model as a GPT model.
- k (int): Parameter for the number of retrievals.
- cuda_visible_devices (str): Specifies which CUDA devices to make visible
- num_gpus (int): Number of GPUs you want to use
```
