import torch
import json
from sentence_transformers import SentenceTransformer
from modelscope import AutoTokenizer, AutoModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType


llm_path = '/root/autodl-tmp/llm-research/meta-llama-3-8b-instruct'
tokenizer = AutoTokenizer.from_pretrained(llm_path)

# 确保 pad_token 已设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型
llm = AutoModelForCausalLM.from_pretrained(
    llm_path,
    device_map="auto",  # 自动分配设备
    torch_dtype=torch.bfloat16
)



vault_path = "/root/autodl-tmp/vault_detail.txt"
Prompt_text = "Identify skills from the Relevant Context that align with the skills presented in the provided job description. **Output format**: ['skill_1_Name'; 'skill_2_Name'; 'skill_3_Name'; 'skill_4_Name'; …]. Job Description: "

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'


print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
vault_content = []

with open(vault_path, "r", encoding='utf-8') as vault_file:
    vault_content = vault_file.readlines()

print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
vault_embeddings = []



model_dir = '/root/autodl-tmp/AI-ModelScope/bge-large-zh-v1.5'
model = SentenceTransformer(model_dir)
for content in vault_content:
    embeddings = model.encode(content, normalize_embeddings=True)
    vault_embeddings.append(embeddings)

print("Converting embeddings to tensor...")
vault_embeddings_tensor = torch.tensor(vault_embeddings) 
print("Embeddings for each line in the vault:")
print(vault_embeddings_tensor)



def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings_tensor, vault_content, top_k=15):
    if vault_embeddings_tensor.nelement() == 0:  # Check if the tensor has any elements
        return []
    input_embedding = model.encode(rewritten_input, normalize_embeddings=True)
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings_tensor)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context




def generate_responses(input_json, output_json):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []

    for item in data:
        #id = item.get("id", "")
        ori_text = item.get("input","")
        rewritten_text = item.get("output", "")
        #小心这里面哦
        relevant_context = get_relevant_context(rewritten_text, vault_embeddings_tensor, vault_content)
        if relevant_context:
            context_str = "\n".join(relevant_context)
            print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
            user_input_with_context = Prompt_text + rewritten_text + "\n\nRelevant Context:\n" + context_str
        else:
            print(CYAN + "No relevant context found." + RESET_COLOR)
            user_input_with_context = rewritten_text
            
        messages = [{"role": "user", "content": user_input_with_context}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 确保输入和模型在同一设备
        model_inputs = tokenizer([text], return_tensors="pt", padding=True)
        model_inputs = {k: v.to(llm.device) for k, v in model_inputs.items()}

        # 生成回答
        generated_ids = llm.generate(
            model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=1024,
            pad_token_id=tokenizer.pad_token_id  # 显式设置 pad_token_id
        )

        # 获取回答部分
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 保存结果
        results.append({
            "input": ori_text,
            "r_text": rewritten_text,
            "output": response
        })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)



# 示例调用
input_json = "/root/autodl-tmp/liepin_sample_mini2.json" # 输入 JSON 文件路径
output_json = "/root/autodl-tmp/trail3_output.json"  # 输出 JSON 文件路径

print("abc")
print(input_json)
print(output_json)
generate_responses(input_json, output_json)

