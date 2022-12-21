import os
from datasets import load_dataset


for split in ['train', 'validation', 'test']:
    for size in ['small', 'medium', 'large']:
        if split == 'train':
            if not os.path.isfile(f"{split}_{size}.jsonl"):
                dataset = load_dataset("taln-ls2n/kpbiomed", size)
                dataset[split].to_json(f"{split}_{size}.jsonl")
        else:
            if not os.path.isfile(f"{split}.jsonl"):
                dataset = load_dataset("taln-ls2n/kpbiomed", size)
                dataset[split].to_json(f"{split}.jsonl")
