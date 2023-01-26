import json
import re

# https://huggingface.co/datasets/loubnabnl/bigcode_csharp/tree/main
with open('c_sharp_data_0003.jsonl', 'r', encoding='utf-8-sig') as f:
    json_list = list(f)

text = ''
counter = 0
with open('c_sharp_data_test.txt', 'w', encoding='utf-8-sig') as f:
    for json_str in json_list:
        counter += 1        
        result = json.loads(json_str)
        text = result['content']
        lines = text.splitlines()
        text2 = ''
        for line in lines:
            if len(line) == 0:
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            text2 += line + '\n'
        print(f"{counter} c_sharp files processed.")
        f.write(text2)