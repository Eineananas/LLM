# LLM fine-tune
#导入环境
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

#将JSON文件转换为CSV文件
df = pd.read_json('./huanhuan.json')
ds = Dataset.from_pandas(df)
ds[:3]
print('--------------------------------------------')
print(ds[:3])
#处理数据集
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct',use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token, tokenizer.pad_token_id, tokenizer.eos_token_id
print('--------------------------------------------')
print(tokenizer.pad_token, tokenizer.pad_token_id, tokenizer.eos_token_id)

def process_func(example):
    MAX_LENGTH = 384
    # 这里需要根据模型的最大输入长度和batch size设置  需要满足：MAX_LENGTH * batch size < max token input
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"user\n\\n{example['instruction'] + example['input']}assistant\\n\\n", add_special_tokens=False)
    response = tokenizer(f"{example['output']}",add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels =[-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
    }
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
print(tokenizer.decode(tokenized_id[0]['input_ids']))
tokenizer.decode(list(filter(lambda x:x!= -100, tokenized_id[1]["labels"])))

#创建模型
import torch
model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()#开启梯度检查点时，要执行该方法
model.dtype
# lora
from peft import LoraConfig, TaskType, get_peft_model
Config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    inference_mode=False,
    #训练模式
    r=8,
    #Lora 秩
    lora_alpha=32,
    # Lora alpha，具体作用参见 Lora 原珲
    lora_dropout=0.1
    # Dropout 比例
)
model = get_peft_model(model, Config)
model.print_trainable_parameters()
#配置训练参数
# 配置训练参数
args = TrainingArguments(
    output_dir="./output/llama3",
    #这是训练日志的保存路径
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = tokenized_id,
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True),
)
trainer.train()
# 保存 LORA 和 tokenizer 结果
peft_model_id="./llama3_lora"
# 这个路径是lora模块的保存路径
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
