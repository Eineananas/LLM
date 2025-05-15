import json
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm  # 进度条
import time

# 加载环境变量
load_dotenv()

# 初始化 Ollama 客户端
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # useless
)

def query_ollama(prompts_json, output_json, model):
    with open(prompts_json, 'r', encoding='utf-8') as f:
        data = f.read().replace('\x00', '')  # 去除无效字符
        data = json.loads(data, strict=False)
    
    results = []
    for item in tqdm(data, desc="Processing prompts"):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        prompt = f"{instruction}\n\n{input_text}"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        if response.choices and len(response.choices) > 0:
            answer = response.choices[0].message.content
            results.append({
                "instruction": instruction,
                "input": input_text,
                "output": answer
            })
            #print(answer)
        else:
            results.append({
                "instruction": instruction,
                "input": input_text,
                "output": "No response found!"
            })
            print("No response found")
            
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# 示例调用
prompts_json = 'C:/Users/b163450/Documents/标注0327/SLF_LB2.json'  # 输入 JSON 文件路径
output_json = "C:/Users/b163450/Documents/标注0327/SLF2_output.json"  # 输出 JSON 文件路径
model = "llama3:8b"
query_ollama(prompts_json, output_json, model)
