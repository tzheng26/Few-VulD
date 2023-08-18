import re
import os
import json, random


json_path = "../Datasets/SARD_4/Program_Samples/CWE_info/sard_info.json"
data_path = "../Datasets/SARD_4/Pre_Program_Samples/symbols"
# save_path = "../Datasets/SARD_4"
# tokenizer_path = "../w2v/SARD_4/tokenizer.pickle"

# 读取数据集 JSON 文件
with open(json_path, "r") as f:
    json = json.load(f)


print("SARD_4's json file's keys in the first larer:")
print(
    json.keys()
)  # ['General', 'API function call', 'Arithmetic expression', 'Array usage', 'Pointer usage']
print()

print("keys of the second layer of SARD_4's json file:")
for k in json.keys():
    print(k)
    print(json[k].keys())
    print()

print("keys of the third layer of SARD_4's json file:")
for k in json.keys():
    print(k)
    for k2 in json[k].keys():
        print(k2)
        print(json[k][k2].keys())
        print()
