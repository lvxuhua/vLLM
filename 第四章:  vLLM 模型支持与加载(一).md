# 第四章 vLLM 模型支持与加载
##本章课程目标
- 掌握vLLM支持的模型
- 了解vLLM支持这些模型的原因
- 掌握模型下载
- 掌握通过vllm加载本地模型
- 掌握模型格式
## 作业
- 注册huggingface账户
- 下载量化模型到/data/model目录下
- 通过vllm 加载下载的模型并把日志保存到/var/log/vllm.log
- 通过curl访问并保存结果到/data/result/curl.txt
  
vLLM 作为高性能大模型推理引擎，核心优势之一是具备广泛的模型兼容性与灵活的加载机制，可适配不同模态、不同架构的主流大模型，同时支持多种加载方式与自定义适配，满足各类部署场景需求。本章将围绕模型实现方式、模型加载方法、支持模型列表及推理模型支持四大核心板块，结合官网规范与实操细节，系统讲解 vLLM 模型支持与加载的全流程。

# 4.1 模型实现方式（Model Implementation）

vLLM 对模型的支持通过两种核心实现方式完成，兼顾推理性能与兼容性，适配不同场景下的模型使用需求，两种方式的核心差异的在于是否为 vLLM 专属优化。

## 4.1.1 vLLM 原生实现

vLLM 原生实现的模型，是专门为 vLLM 推理引擎优化开发的，其核心代码存放于 `vllm/model_executor/models` 目录下。这类模型深度适配 vLLM 的核心优化特性（如 PagedAttention 内存管理、连续批处理、CUDA 图优化等），能最大限度发挥 GPU 性能，实现高吞吐量、低延迟的推理效果，是生产环境中优先选择的实现方式.

原生实现的模型涵盖纯文本、多模态等各类主流架构，后续 4.3 节支持模型列表中，所有标注为“原生支持”的模型均采用该实现方式，无需额外配置即可享受完整优化。

## 4.1.2 Transformers 后端

为兼容更多未被 vLLM 原生支持的模型，vLLM 提供了 Transformers 后端支持功能，可直接调用 Hugging Face Transformers 库中的模型实现，无需对模型进行额外适配。
通过自定义模型实现和注意力后端，在模型加载过程中透明地注入 PagedAttention 支持，从而让任何被支持的 Hugging Face 模型都能获得高性能推理。
目前，Transformers 后端支持的范围明确，覆盖以下核心类别：

- 模态：纯文本模型、嵌入模型、视觉-语言模型（仅支持图像输入，视频输入支持正在规划中）；

- 架构：仅编码器（Encoder-only）、仅解码器（Decoder-only）、混合专家（Mixture-of-Experts, MoE）架构；

- 注意力类型：全注意力（Full Attention）和/或滑动注意力（Sliding Attention）。

可通过极简代码验证模型是否使用 Transformers 后端：

```python
from vllm import LLM
if __name__ == "__main__":
# 替换为模型名称或本地路径
	llm = LLM(model="模型名称/路径", task="generate")
# 打印模型实现类型，若输出以Transformers开头则为Transformers后端
	llm.apply_model(lambda model: print(type(model)))

```

若模型存在 vLLM 原生实现，也可强制使用 Transformers 后端：离线推理时设置 `model_impl="transformers"`；在线服务时添加命令行参数 `--model-impl transformers。

# 4.2 如何加载模型（Loading a Model）

vLLM 提供灵活的模型加载方式，默认支持从 Hugging Face Hub 加载，同时针对中国用户优化了 ModelScope 加载渠道，还支持自定义模型加载，适配在线、离线等不同部署环境，操作简洁且可扩展性强。

## 4.2.1 从 Hugging Face Hub 加载（默认方式）

默认情况下，vLLM 会从 Hugging Face Hub 自动拉取模型，只需指定模型名称即可完成加载，模型下载后会自动缓存至本地，后续加载无需重复下载。

判断模型是否支持该方式加载，核心方法是检查 Hugging Face 模型仓库中的 `config.json` 文件：查看文件中 `"architectures"` 字段的值，若该值在 vLLM 官方支持的架构列表中（详见 4.3 节），则模型理论上可被原生支持，直接加载即可使用[superscript:3]。

实操示例（加载 LLaMA-3-8B-Instruct 模型）：

```python
from vllm import LLM, SamplingParams

# 采样参数配置（按需调整）
sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
# 从 Hugging Face Hub 加载模型（自动缓存）
llm = LLM(model="meta-llama/Llama-3-8B-Instruct", task="generate")
# 推理测试
prompts = ["请介绍 vLLM 的核心优化特性"]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"提示词：{output.prompt}")
    print(f"生成结果：{output.outputs[0].text}\n")

    
##也可以使用hf命令下载模型
hf download HuggingFaceH4/zephyr-7b-beta

hf download HuggingFaceH4/zephyr-7b-beta --cache-dir ./path/to/cache

查询下载模型
# List cached models
hf scan-cache

# Show detailed (verbose) output
hf scan-cache -v

# Specify a custom cache directory
hf scan-cache --dir ~/.cache/huggingface/hub
```

## 4.2.2 从 ModelScope 加载（适配中国用户）

针对中国用户网络环境，vLLM 支持通过 ModelScope 平台加载模型，只需设置环境变量 `VLLM_USE_MODELSCOPE=True`，即可切换至 ModelScope 加载渠道，无需修改其他代码逻辑[superscript:3][superscript:5]。

```
#可以参照ModelScope官网下载模型或者直接加载模型
```



## 4.2.3 加载自定义模型

即使模型不在 vLLM 官方支持列表中，只要满足 Transformers 规范与 vLLM 兼容性要求，即可通过 Transformers 后端在 vLLM 中运行，核心只需设置 `trust_remote_code=True` 即可启用远程代码加载。

自定义模型需满足的核心条件：

1. 符合 Transformers 规范
   
    - 模型目录结构完整，包含 `config.json`、权重文件（如 safetensors、pt 文件）等核心文件；
    
    - `config.json` 文件中必须包含 `auto_map.AutoModel` 配置，用于指定模型类映射[superscript:3]。
    
2. 满足 vLLM 兼容性要求：     
   
   - 自定义逻辑需在基础模型中实现（如 `MyModel`，而非 `MyModelForCausalLM`）；
   
   - 模型初始化时，需将 `kwargs` 参数向下传递至所有子模块（如注意力层、专家层）[superscript:3]；
   
   - 仅编码器模型需在注意力层添加 `is_causal = False` 配置[superscript:3]。

实操示例（加载自定义模型）：

```python
from vllm import LLM
# 加载本地自定义模型，启用远程代码信任
llm = LLM(
    model="/path/to/custom-model",  # 本地自定义模型路径
    task="generate",
    model_impl="transformers",      # 强制使用Transformers后端
    trust_remote_code=True          # 信任远程/本地自定义代码
)
# 验证模型加载
llm.apply_model(lambda model: print(f"模型实现类型：{type(model)}"))

```

# 4.3 支持模型列表（Supported Models List）

这是本章核心内容，vLLM 官方将支持的模型分为纯文本语言模型、多模态语言模型两大类，以下以表格形式详细列出各类模型的核心信息，包括架构、模型系列、HF 示例模型及功能支持矩阵，所有内容均参考 vLLM 官网最新规范[superscript:3]。

## 4.3.1 纯文本语言模型（Text-only Language Models）

纯文本语言模型是 vLLM 支持最完善的类别，分为生成式模型（用于文本生成）和池化模型（用于嵌入、分类等任务）两大类，具体支持列表如下：

### 4.3.1.1 生成式模型（Generative Models）

|架构（Architecture）|模型系列（Models）|Hugging Face 示例模型（Example HF Models）|功能支持矩阵（LoRA/PP/TP/分块预填充/推测解码）|
|---|---|---|---|
|LlamaForCausalLM|LLaMA、LLaMA 2、LLaMA 3|meta-llama/Llama-3-8B-Instruct、meta-llama/Llama-2-7B-Chat|LoRA✅、PP✅、TP✅、分块预填充✅、推测解码✅|
|Qwen2ForCausalLM|Qwen2、Qwen|Qwen/Qwen2-7B-Instruct、Qwen/Qwen-7B-Chat|LoRA✅、PP✅、TP✅、分块预填充✅、推测解码✅|
|ChatGLMModel|ChatGLM、ChatGLM2、ChatGLM3|THUDM/chatglm3-6b、THUDM/chatglm2-6b|LoRA✅、PP✅、TP✅、分块预填充✅、推测解码⚠️（部分支持）|
|MistralForCausalLM|Mistral、Mixtral（MoE）|mistralai/Mistral-7B-Instruct-v0.2、mistralai/Mixtral-8x7B-Instruct-v0.1|LoRA✅、PP✅、TP✅、分块预填充✅、推测解码✅|
|AquilaForCausalLM|Aquila、Aquila2|BAAI/Aquila-7B、BAAI/AquilaChat-7B|LoRA✅、PP✅、TP✅、分块预填充✅、推测解码⚠️（部分支持）|
|BaiChuanForCausalLM|Baichuan、Baichuan2|baichuan-inc/Baichuan2-13B-Chat、baichuan-inc/Baichuan-7B|LoRA✅、PP✅、TP✅、分块预填充✅、推测解码✅|
|BloomForCausalLM|BLOOM、BLOOMZ、BLOOMChat|bigscience/bloom、bigscience/bloomz|LoRA⚠️（部分支持）、PP✅、TP✅、分块预填充✅、推测解码⚠️（部分支持）|
### 4.3.1.2 池化模型（Pooling Models）

池化模型主要用于非生成类任务，如文本嵌入（Embedding）、分类、情感分析、奖励评分等，核心支持列表如下：

|架构（Architecture）|模型系列（Models）|Hugging Face 示例模型（Example HF Models）|功能支持矩阵（LoRA/PP/TP/批量推理）|
|---|---|---|---|
|BertForSequenceClassification|BERT、DistilBERT、RoBERTa|bert-base-uncased、distilbert-base-uncased、roberta-base|LoRA✅、PP✅、TP✅、批量推理✅|
|BgeForSequenceClassification|BGE|BAAI/bge-m3、BAAI/bge-large-en-v1.5|LoRA✅、PP✅、TP✅、批量推理✅|
|ElectraForSequenceClassification|ELECTRA|google/electra-base-discriminator|LoRA⚠️（部分支持）、PP✅、TP✅、批量推理✅|
|RewardModel|奖励模型（如 StarCoderRewardModel）|bigcode/starcoderbase-gpt4all-judge|LoRA❌、PP✅、TP✅、批量推理✅|
## 4.3.2 多模态语言模型（Multimodal Language Models）

vLLM 目前支持主流视觉-语言多模态模型，仅接受图像输入（视频输入支持正在规划中），可实现图像描述生成、图文检索、图像问答等任务，核心支持列表如下：

|架构（Architecture）|模型系列（Models）|Hugging Face 示例模型（Example HF Models）|功能支持矩阵（LoRA/PP/TP/图像输入）|
|---|---|---|---|
|CLIPModel|CLIP|openai/clip-vit-base-patch32、openai/clip-vit-large-patch14|LoRA⚠️（部分支持）、PP✅、TP✅、图像输入✅|
|Blip2ForConditionalGeneration|BLIP-2|Salesforce/BLIP2-OPT-2.7B、Salesforce/BLIP2-Flan-T5-XXL|LoRA✅、PP✅、TP✅、图像输入✅|
|LlavaForConditionalGeneration|LLaVA、LLaVA-1.5、LLaVA-NeXT|liuhaotian/LLaVA-1.5-7B、liuhaotian/LLaVA-NeXT-7B|LoRA✅、PP✅、TP✅、图像输入✅|
|Qwen2VLForConditionalGeneration|Qwen-VL、Qwen3-VL|Qwen/Qwen3-VL-8B-Instruct、Qwen/Qwen-VL-7B|LoRA✅、PP✅、TP✅、图像输入✅|
注：功能支持矩阵中，✅表示完全支持，⚠️表示部分支持，❌表示不支持；PP=流水线并行，TP=张量并行。

# 4.4 推理模型的支持（Reasoning Models）

vLLM 新版文档中，特别新增了对推理模型（Reasoning Models）的专项支持，这类模型专注于逻辑推理、结构化输出、工具调用等场景，典型代表为 DeepSeek R1 系列，核心支持详情如下[superscript:1]：

## 4.4.1 核心支持的推理模型

|模型系列（Models）|所需解析器名称（Parser Name）|Hugging Face 示例模型（Example HF Models）|核心支持功能|
|---|---|---|---|
|DeepSeek R1|deepseek_r1|deepseek-ai/DeepSeek-R1-Distill-Qwen-7B、deepseek-ai/DeepSeek-R1-7B|结构化输出✅、工具调用✅、LoRA✅、TP✅、PP✅|
## 4.4.2 推理模型的特殊配置

加载推理模型时，需注意以下特殊配置，确保充分发挥模型推理能力

- 需启用远程代码信任：设置 `trust_remote_code=True`，因推理模型通常包含自定义推理逻辑；

- 采样参数配置：建议设置 `temperature=0.5~0.7`（推荐 0.6），兼顾推理准确性与多样性；

- 最大上下文长度：根据模型要求设置 `max_model_len`，DeepSeek R1 系列推荐设置为 4096 或 8192；

- 多卡并行：若模型参数较大（如 13B 及以上），可设置 `tensor_parallel_size` 实现多 GPU 张量并行加载。

实操示例（加载 DeepSeek R1 推理模型）：

```python
from vllm import LLM, SamplingParams

# 采样参数配置（适配推理模型）
sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=8192,
    top_p=0.95
)
# 加载 DeepSeek R1 推理模型
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    task="generate",
    trust_remote_code=True,
    tensor_parallel_size=2,  # 2张GPU张量并行
    gpu_memory_utilization=0.8,  # 显存利用率80%
    max_model_len=8192
)
# 推理测试（结构化输出需求）
prompts = ["请基于以下数据，生成一份结构化的分析报告：{数据内容}"]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0].outputs[0].text)

```

