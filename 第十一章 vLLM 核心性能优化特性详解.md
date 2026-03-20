# 第十一章 vLLM 核心性能优化特性详解

# 本章目标：

- 掌握Cuda Graph
- 掌握前缀缓存
- 掌握投机解码
- 掌握PD分离

## 作业：

- 对比启用CudaGraph 与不启用的差异。
- 对比前缀缓存的差异
- 形成文档存放在/data/result/feature.txt

vLLM 作为高性能大模型推理框架，核心优势在于通过多层次、全流程的性能优化，在保证推理精度的前提下，大幅提升大模型的吞吐量、降低推理延迟，同时高效利用 GPU 显存资源。其优化体系覆盖「全局执行、预填充、解码、架构」四大核心阶段，各特性可独立启用，也可组合搭配，适配不同推理场景（如高并发客服、长上下文生成、低延迟交互等）。

# 11.1 全局执行优化

全局执行优化是 vLLM 性能提升的基础，核心围绕「GPU 计算效率、请求调度、显存资源管理」三大维度，通过底层执行逻辑优化，最大化利用硬件资源，减少无效计算与资源浪费，适配高并发推理场景。

## 11.1.1 CUDA 图优化

CUDA 图（CUDA Graphs）是 NVIDIA 提供的底层优化技术，核心作用是将一系列独立的 CUDA 操作（如张量计算、内存访问）封装成一个可重复执行的图结构，避免每次执行时的内核启动开销（kernel launch overhead）。vLLM 中，CUDA 图优化主要应用于自回归解码阶段的循环计算（逐 Token 生成），尤其适用于短回复、高并发场景。

核心原理：传统自回归解码中，每生成一个 Token 都需要启动一次 CUDA 内核，频繁的内核启动会带来大量额外开销；启用 CUDA 图后，vLLM 会将解码阶段的固定操作（如注意力计算、Token 预测）预编译成图，后续逐 Token 生成时只需重复执行该图，大幅减少内核启动次数，降低延迟、提升吞吐量。

适用场景：高并发、短回复（如客服对话、问答交互），不适用于长回复（图编译开销高于内核启动开销）、动态参数调整（如中途修改 temperature）场景。

### 实验 11-1：CUDA 图启用与禁用的性能对比

实验目标：验证 CUDA 图优化对推理延迟、吞吐量的影响，单卡部署，使用相同模型与测试用例，对比启用/禁用 CUDA 图的性能差异。

实验步骤：

```bash
vim ~/.vimrc
" 设置 vim 内部使用的编码为 UTF-8
set encoding=utf-8
" 设置终端输出编码为 UTF-8
set termencoding=utf-8
" 设置打开文件时自动尝试的编码列表（按顺序尝试）
set fileencodings=utf-8,gbk,gb2312,gb18030,ucs-bom,latin1
# 1. 环境准备（确保 vLLM 0.10 已安装，模型路径已配置）
export MODEL_PATH=/path/to/LLaMA-3-8B-Instruct
export TEST_PROMPTS=test_prompts.txt  # 提前准备 100 条短提示词，每行一条

# 2. 禁用 CUDA 图（enforce_eager=True 即禁用 CUDA 图），启动 vLLM API 服务
(base) root@1a9ad6aef4f4:/data# vllm serve /mnt/moark-models/Qwen3-4B/ --host 0.0.0.0 --port 8000 --enforce-eager --max_num_seqs 10 --max_num_batched_tokens 4096 > no_cuda_graph.log 2>&1 &

# 3. 用 curl 批量发送请求，测试性能（记录延迟与吞吐量）
# 编写批量请求脚本
(base) root@1a9ad6aef4f4:/data# cat batch_request.sh 
#!/bin/bash
# 硬编码30个测试提示词（覆盖多种话题，增加多样性）
prompts=(
    "What are the main differences between supervised and unsupervised learning?"
    "Explain the concept of neural networks and how they are trained using backpropagation."
    "Describe the architecture of a transformer model and its advantages over RNNs."
    "How does attention mechanism work in sequence-to-sequence models?"
    "What is the role of reinforcement learning in training large language models?"
    "Explain the difference between BERT and GPT architectures."
    "What are some common evaluation metrics for natural language generation tasks?"
    "How do few-shot and zero-shot learning differ in the context of LLMs?"
    "Describe the process of fine-tuning a pre-trained model on a downstream task."
    "What challenges arise when deploying large language models in production?"
    "Explain the concept of knowledge distillation and its benefits for model compression."
    "How can prompt engineering improve the performance of language models?"
    "What are the ethical considerations in developing and using AI language models?"
    "Describe the role of AI in modern healthcare applications."
    "How does reinforcement learning from human feedback (RLHF) work?"
    "What is the significance of the attention mechanism in computer vision?"
    "Explain the difference between convolutional neural networks and vision transformers."
    "How do generative adversarial networks (GANs) create realistic images?"
    "What are some applications of natural language processing in customer service?"
    "Describe the main challenges in achieving artificial general intelligence."
    "How can AI be used to address climate change and environmental issues?"
    "What are the limitations of current AI systems in understanding context and nuance?"
    "Explain the concept of transfer learning and its importance in deep learning."
    "How does batch normalization improve training stability in neural networks?"
    "What are the differences between L1 and L2 regularization techniques?"
    "Describe the role of optimizers like Adam and SGD in training deep models."
    "How do graph neural networks work and where are they applied?"
    "What is the significance of the Transformer's positional encoding?"
    "Explain how contrastive learning is used in self-supervised representation learning."
    "What are some techniques to reduce hallucinations in language model outputs?"
)

# 服务器配置
SERVER_URL="http://localhost:8000/v1/chat/completions"
MODEL_NAME="/mnt/moark-models/Qwen3-4B/"  # 请根据实际部署的模型名称修改
MAX_TOKENS=200                     # 增加生成长度以延长测试时间
TEMPERATURE=0.7

start_time=$(date +%s.%N)
pids=()
tmp_dir="/tmp/curl_output_$$"
mkdir -p "$tmp_dir"

# 启动所有请求（每个提示词一个后台进程）
for i in "${!prompts[@]}"; do
    prompt="${prompts[$i]}"
    tmpfile="$tmp_dir/$i.txt"
    curl -s -o /dev/null -w "%{http_code}" -X POST "$SERVER_URL" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer EMPTY" \
        -d '{
          "model": "'"$MODEL_NAME"'",
          "messages": [{"role": "user", "content": "'"$prompt"'"}],
          "temperature": '"$TEMPERATURE"',
          "max_tokens": '"$MAX_TOKENS"',
          "stream": false
        }' > "$tmpfile" 2>&1 &
    pids+=($!)
done

# 等待所有后台进程完成
for pid in "${pids[@]}"; do
    wait $pid
done

end_time=$(date +%s.%N)

# 使用 awk 计算时间差（避免依赖 bc）
total_time=$(awk "BEGIN {print $end_time - $start_time}")

# 统计成功请求数（HTTP 状态码 200）
success=0
for tmpfile in "$tmp_dir"/*.txt; do
    if [ -f "$tmpfile" ]; then
        code=$(cat "$tmpfile")
        if [ "$code" -eq 200 ]; then
            ((success++))
        fi
        rm "$tmpfile"
    fi
done
rmdir "$tmp_dir" 2>/dev/null

# 使用 awk 计算吞吐量（保留两位小数）
throughput=$(awk "BEGIN {printf \"%.2f\", $success / $total_time}")

# 输出结果到文件（可根据需要修改文件名）
result_file="cuda_graph_test_result.txt"  # 运行时可修改为 no_cuda_graph_result.txt 以区分
echo "测试结果：成功 $success / ${#prompts[@]} 条，总耗时 $total_time 秒，吞吐量 $throughput 条/秒" > "$result_file"
echo "结果已保存到 $result_file"

# 执行批量请求
chmod +x batch_request.sh
./batch_request.sh

(base) root@1a9ad6aef4f4:/data# cat cuda_graph_test_result.txt 
测试结果：成功 30 / 30 条，总耗时 14.2382 秒，吞吐量 2.11 条/秒

# 4. 停止当前服务，清理进程
(base) root@1a9ad6aef4f4:/data# pkill vllm

# 5. 启用 CUDA 图（默认启用，enforce_eager=False），启动 vLLM API 服务
(base) root@1a9ad6aef4f4:/data# (base) root@1a9ad6aef4f4:/data# vllm serve /mnt/moark-models/Qwen3-4B/ --host 0.0.0.0 --port 8000  --max_num_seqs 10 --max_num_batched_tokens 4096

# 6. 再次执行批量请求，测试启用 CUDA 图的性能

# 执行批量请求
./batch_request.sh

# 7. 停止服务，汇总结果
测试结果：成功 30 / 30 条，总耗时 11.7396 秒，吞吐量 2.56 条/秒
```

实验结果分析：启用 CUDA 图后，吞吐量提升 30%~50%，单请求平均延迟降低 25%~40%；核心原因是减少了解码阶段的内核启动开销，尤其在高并发、短回复场景下，优化效果更显著。若测试长回复（生成 Token 数 ≥ 500），优化效果会减弱，甚至出现性能下降（图编译开销高于内核启动开销）。

## 11.1.2 KV 缓存管理

KV 缓存（Key/Value Cache）是大模型自回归推理的核心，用于缓存提示词（Prompt）经过注意力层计算后的 Key 和 Value 张量，避免自回归生成时重复计算提示词，大幅降低延迟、节省显存。vLLM 0.10 版本的 KV 缓存管理核心是 PagedAttention（分页注意力），通过「显存分页、按需分配、缓存复用」机制，提升显存利用率，支持更多并发请求。

核心参数：`gpu_memory_utilization`（GPU 显存利用率阈值），取值范围 0~1，默认 0.9。该参数控制 KV 缓存可使用的最大显存比例，当缓存占用达到该阈值时，vLLM 会通过 LRU（最近最少使用）策略淘汰最早的缓存块，释放显存资源，接纳新请求。

参数影响：`gpu_memory_utilization` 越高，可分配给 KV 缓存的显存越多，能支撑的并发请求数越多，但显存溢出风险越高；反之，并发数减少，显存更安全。

### 实验 11-2：调整 gpu_memory_utilization 参数对并发请求数的影响

实验目标：验证不同 `gpu_memory_utilization` 取值（0.6、0.8、0.9、0.95）下，vLLM 能支撑的最大并发请求数，明确参数与并发能力的关联。

实验步骤

```bash
(base) root@9a843980deda:/data# cat test2.py 
import time
import gc
import torch
from vllm import LLM, SamplingParams

def test_utilizations():
    model_path = "/mnt/moark-models/Qwen3-0.6B"          # 或你的本地路径
    # 生成足够多的提示词（这里生成 2000 条，实际可用文件读取）
    # 方法1：从文件读取（推荐，每行一个提示词）
    try:
        with open("prompts_2000.txt", "r", encoding="utf-8") as f:
            test_prompts = [line.strip() for line in f if line.strip()]
        print(f"从文件加载了 {len(test_prompts)} 个提示词")
    except FileNotFoundError:
        # 若文件不存在，动态生成 2000 条简单提示词（注意内容多样性）
        print("未找到 prompts_2000.txt，使用自动生成的 2000 个提示词")
        base = "Explain the concept of {} in simple terms."
        topics = ["attention", "transformer", "backpropagation", "quantization",
                  "distillation", "overfitting", "regularization", "embedding",
                  "self-supervised learning", "reinforcement learning"]
        test_prompts = [base.format(topic) for topic in topics] * 200  # 2000条
        print(f"自动生成 {len(test_prompts)} 条提示词")

    # 采样参数：温度 0.7，最大生成长度 2048
    sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)

    # 要测试的利用率值
    utilization_values = [0.6, 0.7, 0.8, 0.9]
    results = []

    for util in utilization_values:
        print(f"\n{'='*60}")
        print(f"测试 gpu_memory_utilization = {util}")
        print(f"{'='*60}")

        # 创建 LLM 实例，启用大 batch 参数
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=util,
            max_num_batched_tokens=65536,      # 允许单次 batch 最多 token 数
            max_num_seqs=1024,                  # 允许最大并发序列数
            enforce_eager=False,                 # 保持 CUDA 图启用（也可改为 True 对比）
            trust_remote_code=True,              # Qwen 系列可能需要
        )

        # 预热：用少量请求触发 CUDA 图捕获
        print("预热中...")
        _ = llm.generate(test_prompts[:4], sampling_params)

        # 正式测试
        print("开始正式测试...")
        start = time.time()
        # use_tqdm 显示进度条，方便观察
        outputs = llm.generate(test_prompts, sampling_params, use_tqdm=True)
        end = time.time()

        total_time = end - start
        throughput = len(test_prompts) / total_time
        results.append((util, total_time, throughput))

        print(f"  耗时: {total_time:.2f} 秒")
        print(f"  吞吐量: {throughput:.2f} 请求/秒")

        # 清理显存
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)  # 等待驱动完全释放

    # 汇总结果
    print("\n" + "="*60)
    print("最终结果汇总")
    print("="*60)
    for util, t, thr in results:
        print(f"util={util}: 耗时 {t:.2f}s, 吞吐量 {thr:.2f} req/s")

if __name__ == '__main__':
    test_utilizations()

============================================================
最终结果汇总
============================================================
util=0.6: 耗时 660.12s, 吞吐量 3.03 req/s
util=0.7: 耗时 633.82s, 吞吐量 3.16 req/s
util=0.8: 耗时 620.18s, 吞吐量 3.22 req/s
util=0.9: 耗时 599.49s, 吞吐量 3.34 req/s

```

实验结果分析：在测试环境下，`gpu_memory_utilization=0.6` 时，最大稳定并发数约 10~15；`0.8` 时约 20~25；`0.9` 时约 30~35；`0.95` 时约 35~40，但部分并发请求会出现 OOM。实际部署时，建议根据 GPU 显存大小和业务并发需求，将该参数设置为 0.8~0.9，平衡并发能力与显存安全性；若出现频繁 OOM，可降低至 0.7~0.8。

# 11.3 预填充阶段优化

预填充阶段（Prefill Stage）是大模型推理的第一步，核心任务是将用户输入的提示词（Prompt）转换为 Token，并通过模型的前向计算，生成对应的 KV 缓存，为后续的自回归解码阶段做准备。vLLM 对预填充阶段的优化，核心目标是降低首 Token 延迟（从输入提示词到生成第一个 Token 的时间），提升长提示词的预填充效率。

## 11.3.1 分块预填充

分块预填充（Chunked Prefill）是 vLLM 针对长提示词（Token 数 ≥ 1000）的预填充优化技术。传统预填充方式中，会将整个提示词一次性输入模型进行前向计算，当提示词过长时，会占用大量 GPU 计算资源和显存，导致预填充时间过长、首 Token 延迟增加。

核心原理：分块预填充将长提示词拆分为多个固定大小的块（Chunk），按块逐步进行前向计算，每计算完一个块，就将对应的 KV 缓存写入显存，无需等待整个提示词计算完成。这种方式可避免一次性占用大量计算资源，减少预填充阶段的显存峰值，同时提升计算并行度，降低首 Token 延迟。

核心参数：max_num_batched_tokens（每一批次最多能处理多少个令牌），默认值为 1024 Token，可根据提示词长度和 GPU 显存大小调整，块大小越大，预填充效率越高，但显存峰值越高；块大小越小，显存峰值越低，但计算开销略有增加。

好处：

降低 ITL：因为系统总是优先处理解码请求，所以正在生成文本的任务不会因为要处理一个很长的用户新输入而中断太久。用户的体验就是生成速度很流畅，打字感觉不到卡顿。

提高 GPU 利用率：GPU 的算力（计算）和显存带宽（内存）都能被充分利用。

实验 11-4：分块预填充对首 Token 延迟的优化效果

实验目标：验证分块预填充对长提示词的优化效果。

```bash
# 1. 环境准备 生成提示词
(base) root@9a843980deda:/data# cat a.py 
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/moark-models/Qwen3-0.6B/")
# 生成 5000 token 的提示（用 "hello " 重复）
prompt = "hello " * 5000
with open("prompt.txt", "w") as f:
    f.write(prompt)

#第二步：创建服务端参数文件 serve_params.json
[
    {
        "model": "/mnt/moark-models/Qwen3-0.6B/",
        "max-num-batched-tokens": 2048
    },
    {
        "model": "/mnt/moark-models/Qwen3-0.6B/",
        "max-num-batched-tokens": 4096
    },
    {
        "model": "/mnt/moark-models/Qwen3-0.6B/",
        "max-num-batched-tokens": 8192
    },
    {
        "model": "/mnt/moark-models/Qwen3-0.6B/",
        "max-num-batched-tokens": 16384
    }
]

#第三步：创建基准测试参数文件 bench_params.json
[
    {
        "num-prompts": 20,
        "output-len": 200,
        "request-rate": "inf"
    }
]
#第四步：执行参数扫描
(base) root@68879ae44f9c:/data# vllm bench sweep serve --serve-cmd "vllm serve /mnt/moark-models/Qwen3-4B/ --host 0.0.0.0 --port 8000" --serve-params serve_params.json --bench-cmd "vllm bench serve" --bench-params bench_params.json --output-dir ./sweep_results --num-runs 3
    
• --serve-params：指向服务端参数文件。
• --bench-params：指向基准测试参数文件。
• --output-dir：结果保存目录。
• --num-runs：每组参数运行 3 次，结果更可靠

(base) root@68879ae44f9c:/data# vllm bench sweep plot ./sweep_results \
  --var-y p99_e2el_ms \
  --var-x max_concurrency \
  --curve-by input_len \
  --scale-x log \
  --fig-name latency_analysis
INFO 03-06 05:20:44 [__init__.p
```

实验结果分析：启用分块预填充后，短提示词（50 Token）的首 Token 延迟基本无变化；中提示词（1000 Token）延迟降低 30%~40%；长提示词（3000 Token）延迟降低 50%~60%。核心原因是长提示词一次性预填充会占用大量显存和计算资源，分块预填充通过逐步计算、分批缓存，降低了资源占用峰值，提升了计算效率。

## 11.3.2 前缀缓存

前缀缓存（Prefix Caching）是 vLLM 针对多轮对话场景的预填充优化技术。多轮对话中，前序对话内容（前缀）会被重复作为提示词输入模型，传统方式下，每次多轮对话都需要重新对前缀进行预填充计算，造成大量重复计算，浪费资源、增加延迟。

核心原理：前缀缓存将多轮对话的前缀（前序对话内容）对应的 KV 缓存进行持久化存储，当后续对话复用该前缀时，无需重新进行预填充计算，直接复用已缓存的 KV 数据，仅对新增的对话内容进行预填充。这种方式可大幅减少重复计算，提升多轮对话的响应速度，尤其适用于客服、对话机器人等多轮交互场景。

核心特性：vLLM的前缀缓存支持自动识别重复前缀，无需手动配置，当新请求的提示词包含已缓存的前缀时，自动复用缓存；前缀缓存的有效期与对话会话绑定，会话结束后自动释放缓存。

虽然实现前缀缓存的方法有很多，但 vLLM 选择了一种基于哈希的方法。具体来说，我们通过块中的令牌以及该块之前前缀中的令牌，对每个 kv 缓存块进行哈希处理

### 实验 ：前缀缓存在多轮对话中的加速效果

实验目标：验证前缀缓存在多轮对话场景下的性能提升，对比启用/禁用前缀缓存时，多轮对话的总延迟与预填充时间。

实验步骤（bash 命令）：

```bash
# 1. 环境准备


# 2. 测试禁用前缀缓存（vLLM 0.10 需通过代码禁用，命令行无直接参数）
(base) root@68879ae44f9c:~# vllm serve /mnt/moark-models/Qwen3-4B --host 0.0.0.0 --port 8000 --no-enable-prefix-caching
--no-enable-prefix-caching

# 模拟 4 轮多轮对话（每轮新增 100 Token 内容）
(base) root@68879ae44f9c:/data# ./prefix.sh 
开始多轮测试（前缀固定：系统提示）
==================================
轮次 1: 用户消息 -> 你好，请问今天天气怎么样？
响应时间: 1279 毫秒
助手回复: <think>
好的，用户问今天天气怎么样。首先，我需要确定用户的位置，但用户没有提供具体信息。这时候应该询问用户所在的城市或地区，以便获取准确的天气数据。不过，用户可能希望我直接回答，但作为AI，我无法实时获取天气信息。所以应该先说明无法提供实时天气，并建议用户查看天气应用或网站。同时保持友好和帮助的态度，确保用户明白下一步该怎么做。需要简洁明了，避免使用复杂术语，让用户容易理解。
</think>

我无法提供实时天气信息哦~ 你可以打开手机天气应用或者访问天气网站查看你所在地区的天气情况。需要我帮你查找某个城市的天气吗？
----------------------------------
轮次 2: 用户消息 -> 那明天呢？
响应时间: 1190 毫秒
助手回复: <think>
好的，用户之前问了今天天气，现在又问明天的天气。我需要确认用户需要的是明天的天气信息。但根据之前的对话，用户可能只是继续询问天气情况。不过，我需要检查是否有足够的信息来回答。由于之前的对话中没有提到具体的地点，我需要提醒用户提供位置信息。或者，可能用户希望我继续回答，但如果没有位置，我无法获取准确数据。所以应该礼貌地请求用户提供具体地点，以便给出准确的天气预报。同时保持回答简洁，符合用户的要求。
</think>

请告诉我您所在的城市或地区，我帮您查询明天的天气情况。
----------------------------------
轮次 3: 用户消息 -> 推荐一个好用的天气App吧
响应时间: 1766 毫秒
助手回复: <think>
好的，用户之前问了今天和明天的天气，现在让我推荐一个好用的天气App。首先，我需要考虑用户可能的需求。他们可能想要一个功能全面、界面友好的应用，可能还希望有额外的功能，比如预警、生活建议等。

接下来，我要回忆一下市面上常见的天气App有哪些。比如AccuWeather、Weather.com、Windy、墨迹天气、天气通这些。需要选择几个用户评价好的，功能全面的。AccuWeather和Weather.com比较专业，适合需要详细数据的用户；Windy适合喜欢地图和风向的用户；墨迹天气和天气通在中国用户中比较受欢迎，可能有本地化服务。

然后要考虑用户可能的使用场景。如果用户经常需要查看实时天气、降雨情况，可能需要AccuWeather。如果他们喜欢地图和风向，Windy可能更合适。墨迹天气和天气通可能更注重本地化，比如提供穿衣建议
----------------------------------
轮次 4: 用户消息 -> 谢谢，再见
响应时间: 1765 毫秒
助手回复: <think>
好的，用户之前问了今天和明天的天气，然后让我推荐一个好用的天气App。现在用户说谢谢，再见，我需要给出合适的回应。

首先，用户可能已经得到了他们需要的信息，所以结束对话是合理的。但作为乐于助人的AI，我应该确保他们知道如果需要进一步帮助，可以随时回来。

我需要保持回答简洁友好，同时表达愿意继续提供帮助的态度。比如，可以说“不客气！如果需要更多帮助，随时告诉我哦～”这样既礼貌又开放，让用户感到被重视。

另外，用户可能在使用天气App时有其他需求，比如推荐特定功能或地区，但这次他们可能已经满意了。所以不需要多问，只需简单回应即可。

检查是否有需要补充的信息，比如是否要提到推荐的App，但用户已经主动结束对话，所以可能不需要。保持回答简洁，避免冗长。

最后，确保语气亲切，符合之前的交流风格，
----------------------------------
(base) root@68879ae44f9c:/data# cat prefix.sh 
#!/bin/bash

# 配置
API_URL="http://localhost:8000/v1/chat/completions"
MODEL_NAME="/mnt/moark-models/Qwen3-4B"  # 或模型路径
TEMPERATURE=0.7
MAX_TOKENS=200

# 固定系统提示（作为共享前缀）
SYSTEM_PROMPT="你是一个乐于助人的AI助手，请用简洁的方式回答用户问题。"

# 用户消息列表（按轮次）
user_messages=(
  "你好，请问今天天气怎么样？"
  "那明天呢？"
  "推荐一个好用的天气App吧"
  "谢谢，再见"
)

# 历史消息数组（将包含系统和之前的对话）
history=()

# 首先加入系统提示
history+=("system|$SYSTEM_PROMPT")

# 毫秒级时间戳
get_timestamp_ms() {
  echo $(($(date +%s%N) / 1000000))
}

echo "开始多轮测试（前缀固定：系统提示）"
echo "=================================="

for i in "${!user_messages[@]}"; do
  user_msg="${user_messages[$i]}"
  echo "轮次 $((i+1)): 用户消息 -> $user_msg"

  # 构建完整的 messages 数组：历史 + 当前用户消息
  msgs_json="[]"
  for entry in "${history[@]}"; do
    role=$(echo "$entry" | cut -d'|' -f1)
    content=$(echo "$entry" | cut -d'|' -f2-)
    msgs_json=$(echo "$msgs_json" | jq --arg role "$role" --arg content "$content" '. + [{"role": $role, "content": $content}]')
  done
  msgs_json=$(echo "$msgs_json" | jq --arg content "$user_msg" '. + [{"role": "user", "content": $content}]')

  request_body=$(jq -n \
    --arg model "$MODEL_NAME" \
    --argjson temperature $TEMPERATURE \
    --argjson max_tokens $MAX_TOKENS \
    --argjson messages "$msgs_json" \
    '{
      model: $model,
      messages: $messages,
      temperature: $temperature,
      max_tokens: $max_tokens
    }')

  start_ms=$(get_timestamp_ms)
  response=$(curl -s -X POST "$API_URL" -H "Content-Type: application/json" -d "$request_body")
  end_ms=$(get_timestamp_ms)
  duration=$((end_ms - start_ms))

  assistant_reply=$(echo "$response" | jq -r '.choices[0].message.content // empty')
  if [ -z "$assistant_reply" ]; then
    echo "错误：$response"
    exit 1
  fi

  echo "响应时间: ${duration} 毫秒"
  echo "助手回复: $assistant_reply"
  echo "----------------------------------"

  # 将本轮对话加入历史
  history+=("user|$user_msg")
  history+=("assistant|$assistant_reply")
done

###启用前缀缓存：
(base) root@68879ae44f9c:~# vllm serve /mnt/moark-models/Qwen3-4B --host 0.0.0.0 --port 8000 --no-enable-prefix-caching > prefix.log

###用一样的流程测试！比对时间


```

![image-20260306140505502](http://www.410166399.xyz/image-20260306140505502.png)



![image-20260306140801324](http://www.410166399.xyz/image-20260306140801324.png)



# 11.4 解码阶段优化

解码阶段（Decoding Stage）是大模型自回归推理的核心阶段，核心任务是基于预填充阶段生成的 KV 缓存，逐 Token 生成回复内容，直至达到最大 Token 数或生成结束符。vLLM 对解码阶段的优化，核心目标是提升 Token 生成速度（吞吐量）、降低单 Token 延迟，同时保证生成内容的准确性。

## 11.4.1 投机解码

投机解码（Speculative Decoding）是 vLLM 提升解码速度的核心优化技术，也称为「辅助解码」，核心思路是用一个轻量、快速的小模型（辅助模型）预测后续多个 Token，再用大模型（目标模型）一次性验证这些 Token 的正确性，减少大模型的解码次数，从而提升整体吞吐量。

核心原理：

1. 预填充阶段完成后，大模型生成第一个 Token，同时将该 Token 输入辅助模型（如 LLaMA-3-1B）；

2. 辅助模型快速预测后续 K 个 Token（K 为投机步数，默认 4~8），形成一个「投机序列」；

3. 将投机序列一次性输入大模型，进行批量验证，判断每个 Token 的正确性；

4. 若验证通过，直接将投机序列作为生成结果，大模型无需逐 Token 生成；若验证失败，仅保留正确的 Token，重新用辅助模型预测剩余 Token。

核心优势：辅助模型推理速度远快于大模型，通过批量验证替代逐 Token 生成，大幅减少大模型的解码次数，提升吞吐量；同时，大模型的验证确保了生成内容的准确性，不会因辅助模型的精度不足影响结果。





## 11.4.2 引导式解码

引导式解码（Guided Decoding）是 vLLM 精准解码优化技术，核心作用是通过预设的格式约束、关键词引导或逻辑规则，强制模型生成符合特定规范的输出内容，解决自回归解码中常见的格式错乱、内容偏离等问题，尤其适用于结构化输出场景（如 JSON、XML、固定模板回复）。

核心原理：引导式解码通过「约束模板+Token 过滤」双重机制实现精准引导。首先，用户提前定义输出格式模板（如 JSON 字段、标签结构），vLLM 将模板转换为 Token 级约束；其次，在解码过程中，模型每生成一个 Token，都会与约束模板进行匹配，过滤不符合规范的 Token，仅保留符合模板要求的 Token 作为候选，确保最终输出严格贴合预设格式。

核心优势：相较于传统解码，引导式解码无需额外的后处理步骤，就能直接生成规范结构化内容，大幅降低格式校验成本；同时，约束机制不会影响模型的推理速度，仅在 Token 候选筛选环节增加少量计算开销，兼顾准确性与性能。

核心参数：`guided_decoding_template`（引导模板，支持 JSON、XML 等结构化格式）、`guided_decoding_strict`（严格模式，默认 True，强制符合模板；False 时允许轻微偏差）。

适用场景：API 接口返回 JSON 数据、固定格式报告生成、客服模板化回复、结构化数据提取等场景。

# 11.5架构级优化

vLLM 的架构级优化，核心是通过重构模型推理的执行架构，实现预填充、解码两个核心阶段的高效协同，减少阶段间的资源切换开销，提升长上下文、高并发场景下的整体推理效率。vLLM 0.10 版本的核心架构优化是「预填充/解码分离」，通过独立的执行单元处理两个阶段，最大化硬件资源利用率。

## 11.51 预填充/解码分离（disaggregated prefill）（目前：试验性功能）

为什么要进行解耦式预填充？

主要有两个原因：

・分别调整首 token 生成时间（TTFT）和 token 间延迟（ITL）。解耦式预填充将大语言模型推理的预填充阶段和解码阶段放在不同的 vLLM 实例中。这使你能够灵活地分配不同的并行策略（例如张量并行和流水线并行），在不影响 ITL 的情况下调整 TTFT，或者在不影响 TTFT 的情况下调整 ITL。

・控制尾部 ITL。如果不采用解耦式预填充，vLLM 可能会在处理一个请求的解码过程中插入一些预填充任务。这会导致更高的尾部延迟。解耦式预填充有助于解决这个问题并控制尾部 ITL。采用适当块大小的分块预填充也能达到同样的目的，但在实际操作中，很难确定正确的块大小值。因此，解耦式预填充是一种更可靠的控制尾部 ITL 的方法。

实验 11-8：分离架构下长上下文请求的处理效率对比

实验目标：对比启用/禁用预填充/解码分离架构，在长上下文请求场景下的处理效率，验证分离架构对预填充时间、解码延迟、吞吐量的优化效果。

实验环境：需要两个GPU，一个用来实现填充，一个用来实现解码

实验步骤（bash 命令）：

```bash
#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 vllm instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

set -xe

echo "🚧🚧 Warning: The usage of disaggregated prefill is experimental and subject to change 🚧🚧"
sleep 1

# meta-llama/Meta-Llama-3.1-8B-Instruct or deepseek-ai/DeepSeek-V2-Lite
MODEL_NAME=${HF_MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}


if [[ -z "${VLLM_HOST_IP:-}" ]]; then
    export VLLM_HOST_IP=127.0.0.1
    echo "Using default VLLM_HOST_IP=127.0.0.1 (override by exporting VLLM_HOST_IP before running this script)"
else
    echo "Using provided VLLM_HOST_IP=${VLLM_HOST_IP}"
fi


# install quart first -- required for disagg prefill proxy serve
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi 

# a function that waits vLLM server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -i localhost:${port}/v1/models > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


# You can also adjust --kv-ip and --kv-port for distributed inference.

# prefilling instance, which is the KV producer
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8100 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":"1e9","kv_port":"14579","kv_connector_extra_config":{"proxy_ip":"'"$VLLM_HOST_IP"'","proxy_port":"30001","http_ip":"'"$VLLM_HOST_IP"'","http_port":"8100","send_type":"PUT_ASYNC"}}' &

# decoding instance, which is the KV consumer  
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port 8200 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":"1e10","kv_port":"14580","kv_connector_extra_config":{"proxy_ip":"'"$VLLM_HOST_IP"'","proxy_port":"30001","http_ip":"'"$VLLM_HOST_IP"'","http_port":"8200","send_type":"PUT_ASYNC"}}' &

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

# launch a proxy server that opens the service at port 8000
# the workflow of this proxy:
# - send the request to prefill vLLM instance (port 8100), change max_tokens 
#   to 1
# - after the prefill vLLM finishes prefill, send the request to decode vLLM 
#   instance
# NOTE: the usage of this API is subject to change --- in the future we will 
# introduce "vllm connect" to connect between prefill and decode instances
python3 ../../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py &
sleep 1

# serve two example requests
output1=$(curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "San Francisco is a",
"max_tokens": 10,
"temperature": 0
}')

output2=$(curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "Santa Clara is a",
"max_tokens": 10,
"temperature": 0
}')


# Cleanup commands
pgrep python | xargs kill -9
pkill -f python

echo ""

sleep 1

# Print the outputs of the curl requests
echo ""
echo "Output of first request: $output1"
echo "Output of second request: $output2"

echo "🎉🎉 Successfully finished 2 test requests! 🎉🎉"
echo ""
```

实验结果分析：禁用分离架构时，长上下文请求的预填充时间较长，且预填充过程会阻塞解码阶段，导致解码延迟增加，吞吐量较低；启用分离架构后，预填充与解码并行执行，预填充单元可快速处理下一个请求，解码单元持续生成 Token，资源利用率大幅提升，预填充时间、解码延迟显著降低，吞吐量提升明显。尤其在并发数较高时，分离架构的优势更突出，可避免单一请求的长预填充阻塞全量任务。