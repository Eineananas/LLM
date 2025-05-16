import pandas as pd
import json

def excel_to_json(excel_file, json_file):
    # 读取Excel文件
    df = pd.read_excel(excel_file)

    # 验证是否存在'JobText'列
    if 'JobText' not in df.columns or 'job_genre' not in df.columns:
        raise ValueError("Excel文件中缺少目标列")

    # 构造JSON列表
    json_list = []
    for _, row in df.iterrows():
        json_entry = {
            "instruction": "Based on the job description provided in the input, categorize the job into one of the following categories and output the corresponding number: 11 Management Occupations\n13 Business and Financial Operations Occupations\n15 Computer and Mathematical Occupations\n17 Architecture and Engineering Occupations\n19 Life, Physical, and Social Science Occupations\n21 Community and Social Service Occupations\n23 Legal Occupations\n25 Educational Instruction and Library Occupations\n27 Arts, Design, Entertainment, Sports, and Media Occupations\n29 Healthcare Practitioners and Technical Occupations\n31 Healthcare Support Occupations\n33 Protective Service Occupations\n35 Food Preparation and Serving Related Occupations\n37 Building and Grounds Cleaning and Maintenance Occupations\n39 Personal Care and Service Occupations\n41 Sales and Related Occupations\n43 Office and Administrative Support Occupations\n47 Construction and Extraction Occupations\n49 Installation, Maintenance, and Repair Occupations\n51 Production Occupations\n53 Transportation and Material Moving Occupations\n",
            "input": row['JobText'],
            "output": row['job_genre']
        }
        json_list.append(json_entry)

    # 将JSON列表写入文件
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_list, f, ensure_ascii=False, indent=4)

# 使用示例

excel_file = "E:/Fine_tune_LLM_training_set/training_1300.xlsx"  # 替换为你的Excel文件路径
json_file = 'output.json'  # 替换为你想要生成的JSON文件路径
excel_to_json(excel_file, json_file)
