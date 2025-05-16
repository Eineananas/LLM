import pandas as pd
import json

def json_to_excel(json_file, excel_file):
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 提取JobText
    job_texts = [entry['input'] for entry in json_data]

    # 创建DataFrame
    df = pd.DataFrame(job_texts, columns=['JobText'])

    # 将DataFrame写入Excel文件
    df.to_excel(excel_file, index=False)

# 使用示例
json_file = "D:/Fine_tune_LLM_training_set/jan-liepin-english/ask_liepin_eng2.json"  
excel_file = 'D:/Fine_tune_LLM_training_set/jan-liepin-english/ask_liepin_eng2.xlsx' 
json_to_excel(json_file, excel_file)
