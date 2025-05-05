import torch
from modelscope import snapshot_download
import os
# Download the specific Llama model
model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',cache_dir='/root/autodl-tmp',revision='master')
# Deepseek
model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct',cache_dir='/root/autodl-tmp',revision='master')
# Llama
# You can also download at a specific directory manually
# The cache_dir need to be consistant with the deployment code

# modelscope是一个网站，类似于中国版的hugging face
# snapshot_download 是这个网站提供的一种下载大模型的方法
# deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   还有      llm-research/meta-llama-3-8b-instruct 都是这个网站上的模型名称
# 也可以换成别的魔塔社区里的其他模型
'''
两个开源社区
在我们下载大模型之前，先来介绍两个重要的开源社区!
HuggingFace
Modelscope(魔搭社区)
HuggingFace 是一家成立于纽约的 AI 研究公司，以其开源项目 Transformers 库而闻名.
该库聚焦于自然语言处理(NLP)和机器学习，并支持超过 100 种语言的模型。HuggingFace 强调社区协作，致力于使 AI 更加民主化，为研究人员和开发者提供强大的工具，以推动人工智能技术的进步和应用。
Modelscope(魔搭社区)是由中国的科技巨头阿里巴巴集团旗下的阿里云推出的一个开源平台。
该平台专注于提供各种 AI 模型，包括但不限于自然语言处理、计算机视觉和音频处理。ModelScope 旨在简化 AI 模型的开发和部署过程，使技术更加透明和容易访问，特别是为中国的开发者和研究机构提供支持。
这两个平台可以简单理解为开源大模型的仓库。从这些平台，我们可以下载到各种开源的大模型。
他们的区别可以类比于 qithub 和 gitee 的区别:Huggingface 是国际上的平台，而 Modelscope 则是国内的平台。
'''
