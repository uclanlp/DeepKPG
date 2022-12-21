import sys

if len(sys.argv) != 2:
    print("Usage: parse_result_to_csv.py results_txt_file")

raw_file = sys.argv[1]
out_file = raw_file.replace('.txt', '') + '.csv'

names = ["micro_at_5", "macro_at_5", "micro_at_m", "macro_at_m"]
stats = [[], [], [], []]

with open(raw_file) as in_f:
    cur_idx = 0 
    for line in in_f.readlines():
        if(line[:5] == "Micro" or line[:5] == "Macro"):
            cur_stats = line.split("\t")[-3:]
            cur_stats = [float(x.split("=")[-1]) for x in cur_stats]
            stats[cur_idx].extend(cur_stats)
            cur_idx = (cur_idx+1) % 4

print(stats)

with open(out_file, 'w') as out_f:
    out_f.write("name,all_precision,all_recall,all_f1,present_precision,present_recall,present_f1,absent_precision,absent_recall,absent_f1\n")
    for i in range(4):
        out_f.write(names[i] + ',' + ','.join([str(x) for x in stats[i]]) + '\n')
