from transformers import Blip2Model
import torch
import os


def download_hf_model(hf_model_name):
    name = hf_model_name.split('/')[0] if '/' not in hf_model_name else hf_model_name.split('/')[1]
    model = Blip2Model.from_pretrained(hf_model_name, torch_dtype=torch.float16)

    model_dir = f'./{name}'

    model_parts = [
        'pytorch_model-00001-of-00002.bin',
        'pytorch_model-00002-of-00002.bin'
    ]
    # Load each part of the state dictionary
    state_dicts = [torch.load(os.path.join(model_dir, part)) for part in model_parts]

    # Concatenate the state dictionaries
    full_state_dict = {}
    for state_dict in state_dicts:
        full_state_dict.update(state_dict)

    model.save_pretrained(model_dir, safe_serialization=False, max_shard_size='20GB')


download_hf_model('Salesforce/blip2-opt-2.7b')
