import os
import sys
from datasets import load_dataset


if sys.argv[1] == 'ldkp3k':
    for split in ['train', 'validation', 'test']:
        for size in ['small', 'medium', 'large']:
            if not os.path.isfile(f"ldkp3k_{size}_{split}.jsonl"):
                dataset = load_dataset("midas/ldkp3k", size)
                dataset[split].to_json(f"ldkp3k_{size}_{split}.jsonl")

elif sys.argv[1] == 'ldkp10k':
    for split in ['train', 'validation', 'test']:
        for size in ['small', 'medium', 'large']:
            if not os.path.isfile(f"ldkp10k_{size}_{split}.jsonl"):
                dataset = load_dataset("midas/ldkp10k", size)
                dataset[split].to_json(f"ldkp10k_{size}_{split}.jsonl")

else:
    print("Dataset name not recognized. Exiting...")            
