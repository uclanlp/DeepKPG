import sys
import json

if len(sys.argv) != 3:
    print("Usage: parse_result_to_csv.py name eval_results.jsonl")
    exit()

name = sys.argv[1].replace('eval_results.jsonl', '').replace(',', '')
raw_file = sys.argv[2]
out_file = raw_file.replace('.jsonl', '') + '.csv'

with open(raw_file) as in_f:
    data = json.load(in_f)
    print(data)
    
items = [(str(k), str(v)) for k, v in data.items() if not(type(v) == dict)]
data = {k: v for k, v in data.items() if type(v) == dict}
items += [(str(k), str(v)) for x in data.values() for k, v in x.items()]
items.sort(key=(lambda x: x[0]))
items = [('name', name)] + items
print(items)

with open(out_file, 'w') as out_f:
    print(','.join([x[0] for x in items]), file=out_f)
    print(','.join([x[1] for x in items]), file=out_f)
