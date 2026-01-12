# Random functions to avoid clogging up the notebooks

import json

def count_parameters(model):
    """Detailed parameter count breakdown"""
    total = 0
    breakdown = {}
    
    for name, param in model.named_parameters():
        params = param.numel()
        total += params
        
        # Categorize
        if 'embed' in name:
            breakdown['embeddings'] = breakdown.get('embeddings', 0) + params
        elif 'attn' in name:
            breakdown['attention'] = breakdown.get('attention', 0) + params
        elif 'mlp' in name or 'swiglu' in name:
            breakdown['mlp'] = breakdown.get('mlp', 0) + params
        elif 'ln' in name or 'norm' in name:
            breakdown['norms'] = breakdown.get('norms', 0) + params
        elif 'lm_head' in name:
            # Check if tied
            if param.data_ptr() == model.embed.weight.data_ptr():
                breakdown['lm_head'] = 0  # Tied, don't double count
            else:
                breakdown['lm_head'] = params
        else:
            breakdown['other'] = breakdown.get('other', 0) + params
    
    print("Parameter Breakdown:")
    print("=" * 50)
    for key, value in breakdown.items():
        print(f"{key:20s}: {value:>12,} ({value/total*100:>5.2f}%)")
    print("=" * 50)
    print(f"{'TOTAL':20s}: {total:>12,}")
    
    return total, breakdown


def strip_compile_prefix(state_dict, prefix="_orig_mod."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_synthetic_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def clean_columns(ds):
    # 'the-stack' uses 'content', others use 'text' or 'line'
    if 'content' in ds.column_names:
        ds = ds.rename_column('content', 'text')
    elif 'line' in ds.column_names:
        ds = ds.rename_column('line', 'text')
    # sft -> text
    elif 'question' in ds.column_names and 'answer' in ds.column_names:
        def merge_qa(example):
            example['text'] = f"Question: {example['question']}\nAnswer: {example['answer']}"
            return example
        ds = ds.map(merge_qa)
    
    # Check if 'text' exists, otherwise throw error
    if 'text' not in ds.column_names:
        raise ValueError(f"Dataset has missing 'text' column. Found: {ds.column_names}")
        
    return ds.select_columns(['text'])

