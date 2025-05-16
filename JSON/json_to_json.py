import pandas as pd
import json

def json_to_json(input_file, output_file):
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 提取Skill Name 和 Skill Description
    skill_name = [entry['Skill Name'] for entry in json_data]
    skill_des = [entry['Skill Description'] for entry in json_data]

    # 创建Skill ID
    skill_id = [str(i + 1) for i in range(len(skill_name))]  # 生成从1开始的顺序数列，并转为字符串

    # 创建DataFrame
    df = pd.DataFrame({'Skill ID': skill_id, 'Skill Name': skill_name, 'Skill Description': skill_des})

    # 将DataFrame写入JSON文件
    json_list = []
    for _, row in df.iterrows():
        json_entry = {
            "Skill ID": row['Skill ID'],
            "Skill Name": row['Skill Name'],
            "Skill Description": row['Skill Description']
        }
        json_list.append(json_entry)

    # 将JSON列表写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_list, f, ensure_ascii=False, indent=4)

# 使用示例
input_file = "C:/Users/b163450/Desktop/Skill_EKB.json"
output_file = "C:/Users/b163450/Desktop/Skill_EKB2.json"  # 替换为你的JSON文件路径
json_to_json(input_file, output_file)
