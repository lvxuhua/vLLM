# 第九章 vLLM 性能基准测试与调优

# 9.1 基准测试概述

在 vLLM 推理部署中，基准测试并非可选步骤，而是保障服务稳定性、优化资源利用率的核心前提。本节将明确基准测试的价值、核心工具及环境准备要点，为后续测试实操奠定基础。

## 9.1.1 为什么需要基准测试

vLLM 作为高性能推理引擎，其性能表现受硬件配置（GPU型号、显存大小）、模型规格（参数规模、模态类型）、参数配置（批处理、并行策略）、业务场景（并发量、请求类型）等多因素影响，基准测试的核心价值在于“量化性能、定位瓶颈、验证优化”，具体体现在三个方面：

- 量化性能基线：明确当前软硬件环境下，vLLM 的延迟（首字延迟 TTFT、字间延迟 ITL）、吞吐量（tokens/s）、并发能力等核心指标，建立性能基准，为后续调优提供参考；

- 定位性能瓶颈：通过测试不同参数、不同场景下的性能差异，定位瓶颈所在（如 GPU 显存不足、批处理参数不合理、跨卡通信开销过大等），避免盲目调优；

- 验证调优效果：调优参数（如 max_num_batched_tokens、tensor_parallel_size）后，通过基准测试验证调优是否有效，确保优化后的性能满足业务需求（如实时聊天场景的低延迟要求、批量推理场景的高吞吐量要求）。

核心结论：基准测试是“感知性能、优化性能、保障性能”的核心手段，无论是初次部署还是后续迭代优化，都需通过基准测试提供数据支撑。

## 9.1.2 vLLM Bench 工具介绍

vLLM 官方提供了专用的基准测试工具 `vllm bench`，集成于 vLLM 核心包中，无需额外安装，支持延迟测试、吞吐量测试、在线服务测试三大核心场景，同时提供参数扫描、结果可视化等高级功能，适配不同测试需求，核心特点如下（严格遵循官方文档定义）：

- 轻量集成：与 vLLM 核心代码同步更新，安装 vLLM 后即可直接使用，无需额外依赖，依托 vLLM 原生优化（如 PagedAttention、连续批处理）实现精准测试；

- 场景全面：覆盖离线推理（延迟、吞吐量）、在线服务（模拟高并发请求）两大核心场景，兼容单卡/多卡、单模型/多模型、文本/多模态等各类部署场景；

- 参数灵活：支持自定义模型、硬件、采样参数、批处理参数等，可精准模拟真实业务场景（如实时对话的短输入、批量推理的长输入）；

- 结果精准：自动统计核心性能指标，生成详细报告，支持导出与可视化，便于分析与对比，数据精度贴合生产环境实际表现；

- 高级扩展：支持参数扫描（批量测试不同参数组合）、结果合并、自定义数据集等，提升测试效率，适配大规模性能调优场景。

vllm bench 核心命令分类：基础测试命令（latency/throughput/serve）、高级命令（sweep serve/sweep plot/sweep merge），后续章节将逐一详解其使用方法。



# 9.2 基础基准测试命令

vllm bench 提供三大基础基准测试命令，分别对应延迟测试、吞吐量测试、在线服务测试，覆盖离线推理与在线服务两大核心场景。所有命令均支持自定义模型、输入输出配置、硬件参数，严格遵循官方文档语法，以下逐一详解。

## 9.2.1 延迟基准测试 (vllm bench latency)

延迟基准测试核心目标是评估 vLLM 推理的延迟性能，重点统计首字延迟（TTFT，Time To First Token）、字间延迟（ITL，Inter Token Latency），适用于对延迟敏感的场景（如实时对话）。命令语法、参数、示例均遵循 vLLM 官方文档。

### 9.2.1.1 参数详解

核心参数（按常用程度排序）：

- `--model`：必填，指定测试模型路径（本地路径或 Hugging Face Hub 模型名，如 `meta-llama/Llama-3-8B-Instruct`），支持所有 vLLM 兼容模型；

- `--input-len`：必填，输入prompt的长度（token数），默认值为 128，可根据业务场景调整（如短输入 64、长输入 512）；

- `--output-len`：必填，输出token的长度（token数），默认值为 128，模拟真实输出场景；

- `--batch-size`：可选，批处理大小，默认值为 8，支持批量推理延迟测试，取值范围 1~1024（需匹配 GPU 显存）；

- `--num-iters`：可选，测试迭代次数，默认值为 10，用于消除偶然误差，建议取值 ≥ 5，迭代次数越多，结果越稳定；

- `--tensor-parallel-size`：可选，张量并行度，默认值为 1（单卡），多卡测试时设置为 GPU 数量（如 2、4）；

- `--max-num-batched-tokens`：可选，最大批处理token数，默认值为 2048，用于控制 GPU 显存占用，避免显存溢出；

- `--quantization`：可选，模型量化方式，支持 GPTQ、AWQ、INT4、INT8、FP8（需模型支持），如 `--quantization gptq`，量化可降低显存占用；

- `--output-json`：可选，测试结果输出路径，默认输出到控制台，指定路径后将生成 JSON 格式报告。

### 9.2.1.2 使用示例与输出解读

#### 使用示例:

```bash
(base) root@9c74b76a6bea:/data# vllm bench latency --model /mnt/moark-models/Qwen3-0.6B/ --input-len 128 --output-len 64 --batch-size 4 --num-iters 4 --output-json latency_test_result.json
```

#### 输出解读：

```json
(base) root@9c74b76a6bea:/data# cat latency_test_result.json 
{
    "avg_latency": 0.2438490237109363,
    "latencies": [
        0.24423139728605747,
        0.2435283400118351,
        0.24354064650833607,
        0.2440957110375166
    ],
    "percentiles": {
        "10": 0.24353203196078538,
        "25": 0.24353756988421082,
        "50": 0.24381817877292633,
        "75": 0.2441296325996518,
        "90": 0.24419069141149521,
        "99": 0.24422732669860125
    }
 这个 JSON 文件是 vLLM 延迟基准测试的输出结果
 各字段含义如下：
  • avg_latency  所有请求延迟的算术平均值，单位为秒（此处约为 0.244 秒）。
  • latencies  一个数组，记录了每次请求的延迟值（单位与 avg_latency 相同）。数组长度等于测试的总请求次数（此处为 4 次）。
  • percentiles  一个对象，表示延迟的百分位分布。每个键（如 "10"）代表百分位数，对应的值表示有该百分比的请求延迟小于或等于此数值（单位秒）。
  ◦ "10"：10% 的请求延迟 ≤ 0.24353 秒
  ◦ "50"：50% 的请求延迟 ≤ 0.24382 秒（即中位数）
  ◦ "99"：99% 的请求延迟 ≤ 0.24423 秒
```

关键解读：

- TTFT（首字延迟）：从输入prompt到生成第一个token的时间，反映模型的响应速度，越低越好，实时场景建议 ≤ 100ms；

- ITL（字间延迟）：生成两个连续token的时间间隔，反映模型的生成流畅度，越低越好，理想值 ≤ 5ms；

- 显存占用：需低于 GPU 总显存的 80%，避免显存溢出导致测试失败；若显存不足，可降低 batch-size 或启用量化。

## 9.2.2 吞吐量基准测试 (vllm bench throughput)

吞吐量基准测试核心目标是评估 vLLM 推理的吞吐量性能，重点统计每秒生成token数（tokens/s）、每GPU每秒生成token数（tokens/s/GPU），适用于批量推理场景（如文本生成、数据处理）。命令语法、参数、示例均遵循 vLLM 官方文档。

### 9.2.2.1 参数详解

核心参数（官方文档明确支持，按常用程度排序）：

- `--model`：必填，指定测试模型路径（本地路径或 Hugging Face Hub 模型名），与 latency 命令一致；
- `--num-prompts`：必填，测试的prompt总数，默认值为 100，建议取值 ≥ 50，确保结果具有统计意义；
- `--dataset`：可选，测试数据集路径，默认使用随机生成的prompt，支持自定义数据集（如 ShareGPT 格式），格式为 JSONL；
- `--input-len`：可选，输入prompt长度（token数），默认值为 128，若指定 dataset 则自动读取数据集的输入长度；
- `--output-len`：可选，输出token长度（token数），默认值为 64，若指定 dataset 则自动读取数据集的输出长度；
- `--tensor-parallel-size`：可选，张量并行度，默认值为 1，多卡测试时设置为 GPU 数量；
- `--max-num-batched-tokens`：可选，最大批处理token数，默认值为 4096，影响吞吐量，需根据 GPU 显存调整；
- `--output-json`：可选，测试结果输出路径，默认输出到控制台，生成 JSON 格式报告。

补充说明（官方文档重点提示）：若使用 `--generation-config` 参数，需指定 vLLM 配置文件，仅覆盖默认参数，优先级高于命令行参数。

### 9.2.2.2 使用示例与输出解读

#### 使用示例:

```bash
(base) root@9c74b76a6bea:/data# vllm bench throughput  --model /mnt/moark-models/Qwen3-0.6B/ --num-prompts 200  --max-num-batched-tokens 8192   --output-json throughput_test_result.json
```

#### 输出解读（官方标准输出格式，简化版）：

```json
(base) root@9c74b76a6bea:/data# cat throughput_test_result.json 
{
    "elapsed_time": 5.0267898347228765,
    "num_requests": 200,
    "total_num_tokens": 230400,
    "requests_per_second": 39.78682351477817,
    "tokens_per_second": 45834.420689024446
}

详细说明

    elapsed_time：反映了系统完成这批请求所需的总时间，是计算其他吞吐量指标的基础。
    num_requests：测试规模，越大结果越稳定。
    total_num_tokens：此值取决于每个请求的输出长度（--output-len）和输入长度（--input-len 或数据集）。例如，若平均每个请求生成 1152 个 token，则 200 个请求总 token 数为 200 × 1152 = 230400。
    requests_per_second：直观衡量系统并发处理能力，越高说明系统能同时服务更多请求。
    tokens_per_second：衡量系统生成内容的速度，对于长文本生成场景尤为重要。

关键解读：
吞吐量（tokens/s）：核心指标，反映模型单位时间内的生成能力，越高越好，批量场景优先关注该指标；
```

![image-20260308105042722](http://www.410166399.xyz/image-20260308105042722.png)

关键解读：

- 吞吐量（tokens/s）：核心指标，反映模型单位时间内的生成能力，越高越好，批量场景优先关注该指标；

  

## 9.2.3 在线服务基准测试 (vllm bench serve)

在线服务基准测试核心目标是模拟真实在线服务场景，评估 vLLM 服务的并发处理能力，重点统计吞吐量、延迟分布（TTFT/ITL/TPOT），支持测试本地 vLLM 服务、兼容 OpenAI API 的服务。

vllm bench serve 的核心作用就是模拟不同客户端负载场景，通过调整客户端侧的请求参数（如并发数、请求速率、输入/输出长度、数据集类型等），向一个已经部署的 vLLM 服务发送请求流，从而测试服务在不同压力下的响应性能（包括吞吐量、延迟分布、错误率等）。

它主要关注的是服务端在不同负载下的表现，而不是测试客户端本身的变化。你可以通过 --request-rate、--max-concurrency、--num-prompts 等参数来控制客户端的行为，观察服务端在这些场景下的性能指标（如 TTFT、TPOT、吞吐量等）。这些测试结果能帮助你评估服务的容量、稳定性，并为生产环境的资源配置提供依据。

### 9.2.3.1 参数详解

核心参数（官方文档明确支持，按常用程度排序）：

- `--backend`：必填，服务后端类型，支持 `vllm`（本地 vLLM 服务）、`openai`（兼容 OpenAI API 的服务），默认值为`vllm`；

- `--base-url`：必填，服务基础地址，本地 vLLM 服务默认 `http://localhost:8000/v1`，OpenAI API 服务默认 `https://api.openai.com/v1`；

- `--model`：必填，测试模型名，需与服务端部署的模型一致（如`meta-llama/Llama-3-8B-Instruct`）；

- `--sharegpt-output-len`：可选，ShareGPT 格式数据集的输出长度（token数），默认值为 64，仅当使用 ShareGPT 数据集时生效；

- `--num-prompts`：可选，测试的prompt总数，默认值为 100，建议取值 ≥ 50；

- `--max-concurrency`：可选，并发请求数，默认值为 10，模拟高并发场景，取值范围 1~100；

- `--dataset`：可选，测试数据集路径，支持 ShareGPT 格式（JSONL），默认使用随机生成的prompt；

- `--input-len`：可选，输入prompt长度（token数），默认值为 128，若指定 dataset 则自动读取；

- `--output-path`：可选，测试结果输出路径，默认输出到控制台，生成 JSON 格式报告。

补充说明（官方文档重点提示）：测试前需确保服务端已启动（本地 vLLM 服务启动命令：`vllm serve --model 模型名 --port 8000`），且服务状态正常。

### 9.2.3.2 使用示例

#### 示例1：测试本地 vLLM 服务（官方推荐示例）

```bash
# 先启动本地 vLLM 服务（后台运行）先在一个终端启动
(base) root@9c74b76a6bea:/data# vllm serve /mnt/moark-models/Qwen3-0.6B/ --host 0.0.0.0 --port 8000
###想测试什么样的参数在vllm serve 加入参数即可

# 执行在线服务基准测试
(base) root@9c74b76a6bea:/data# vllm bench serve --backend vllm --base-url http://localhost:8000 --model /mnt/moark-models/Qwen3-0.6B/ --num-prompts 150   --max-concurrency 50  --sharegpt-output-len 64

============ Serving Benchmark Result ============
Successful requests:                     150       
Failed requests:                         0         
Maximum request concurrency:             50        
Benchmark duration (s):                  9.02      
Total input tokens:                      153600    
Total generated tokens:                  19200     
Request throughput (req/s):              16.63     
Output token throughput (tok/s):         2128.93   
Peak output token throughput (tok/s):    2800.00   
Peak concurrent requests:                92.00     
Total token throughput (tok/s):          19160.34  
---------------Time to First Token----------------
Mean TTFT (ms):                          294.50    
Median TTFT (ms):                        162.52    
P99 TTFT (ms):                           1027.12   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          21.03     
Median TPOT (ms):                        21.83     
P99 TPOT (ms):                           22.20     
---------------Inter-token Latency----------------
Mean ITL (ms):                           21.03     
Median ITL (ms):                         17.91     
P99 ITL (ms):                            43.94


总体统计
指标	值	解释
Successful requests	150	成功完成的请求数，本例中所有150个请求都成功。
Failed requests	0	失败的请求数（如超时、服务端错误等），0表示测试期间服务稳定。
Maximum request concurrency	50	测试设置的最大并发请求数（由 --max-concurrency 指定）。
Benchmark duration (s)	9.02	整个基准测试运行的总时长（从第一个请求发送到最后一个请求完成）。
Total input tokens	153600	所有请求的输入 token 总数。本例中平均每个请求输入 153600/150 = 1024 tokens。
Total generated tokens	19200	所有请求生成的输出 token 总数。平均每个请求输出 19200/150 = 128 tokens。
Request throughput (req/s)	16.63	每秒处理的请求数，即 总请求数 / 总耗时。衡量系统处理并发请求的能力。
Output token throughput (tok/s)	2128.93	每秒生成的输出 token 数，即 总生成token数 / 总耗时。衡量系统的生成速度。
Peak output token throughput (tok/s)	2800.00	在测试期间观察到的最高瞬时输出 token 生成速率。反映了系统的峰值生成能力。
Peak concurrent requests	92.00	测试期间同时处理的请求数最大值（可能超过设置的并发限制，因为请求会排队或重叠）。
Total token throughput (tok/s)	19160.34	每秒处理的总 token 数（输入+输出），即 (总输入token数 + 总输出token数) / 总耗时。


TTFT 指从客户端发送请求到接收到第一个输出 token 的时间。它直接影响用户体验（尤其是交互式应用），是衡量服务响应速度的关键指标。
指标	值 (ms)	解释
Mean TTFT	294.50	所有请求 TTFT 的平均值。
Median TTFT	162.52	所有请求 TTFT 的中位数，即 50% 的请求 TTFT 小于此值。
P99 TTFT	1027.12	99% 的请求 TTFT 小于此值。代表极端情况下的响应延迟，用于评估系统的尾部延迟稳定性。

本例中，平均 TTFT 为 294ms，但 P99 高达 1027ms，说明存在少数请求响应较慢，可能受到资源争抢或调度影响。
每输出Token时间（Time per Output Token, TPOT）

TPOT 指从开始生成第一个 token 之后，每个后续输出 token 的平均生成时间。它反映了模型在解码阶段的生成速度，影响整体吞吐量和用户感知的流畅度。
指标	值 (ms)	解释
Mean TPOT	21.03	所有请求的平均 TPOT。
Median TPOT	21.83	TPOT 的中位数。
P99 TPOT	22.20	TPOT 的 99% 分位数。

TPOT 值较小且稳定（P99 仅略高于均值），表明解码阶段非常稳定，没有明显波动。
Token间延迟（Inter-token Latency, ITL）

ITL 与 TPOT 类似，但严格定义为连续两个输出 token 之间的时间间隔。在某些基准测试工具中，ITL 可能直接取 TPOT 的值（因为生成每个 token 的时间间隔就是 TPOT）。
本例中，ITL 的统计值与 TPOT 非常接近，但注意 ITL 的中位数和 P99 与 TPOT 略有差异，这是因为计算方式可能略有不同（ITL 更关注相邻 token 之间的间隔，而 TPOT 通常指平均每个 token 的时间，包括首 token 后的所有 token）。
此处 ITL 的 P99 较高（43.9ms），说明存在少数 token 生成间隔突增，可能由系统调度波动引起。
指标	值 (ms)	解释
Mean ITL	21.03	所有 token 间延迟的平均值。
Median ITL	17.91	ITL 的中位数。
P99 ITL	43.90	ITL 的 99% 分位数。

综合解读
 高吞吐与低延迟：系统在 9 秒内处理了 150 个请求，平均请求吞吐量 16.63 req/s，输出 token 吞吐量 2128.93 tok/s，表现良好。
 首字延迟分布不均：虽然平均 TTFT 仅 294ms，但 P99 高达 1027ms，表明存在明显的长尾延迟。可能需要检查是否由于并发请求突增导致部 分请求排队，或模型预填充阶段计算负担不均衡。
 解码阶段稳定：TPOT 的均值和 P99 非常接近（21ms 左右），说明解码阶段处理稳定，没有明显波动。但 ITL 的 P99 稍高（43.9ms），提示偶尔存在 token 生成间隔突增，可能与系统资源瞬时抢占有关。
并发能力：峰值并发请求数达到 92，超过了设置的 max_concurrency=50，说明请求在服务端排队或同时处理的数量超过了客户端的并发限制，反映了服务端实际的并发处理能力。
这些指标综合反映了模型服务在本次测试条件下的性能特征，可根据业务需求（如对首字延迟敏感还是对吞吐量敏感）进行针对性调优。
This response is AI-generated, for reference only.


关键解读：
- 吞吐量（tokens/s、requests/s）：反映在线服务的并发处理能力，越高越好，高并发场景需重点关注；
- 延迟分布（P50/P90/P99）：P99 延迟是核心指标，反映极端场景下的延迟表现，在线服务建议 P99 TTFT ≤ 200ms、P99 ITL ≤ 10ms；
- TPOT（每输出token耗时）：补充 ITL 指标，更精准反映在线场景下的token生成效率，尤其适用于多模态模型测试（如 Qwen2-VL），需注意多模态场景下 TPOT 会随请求率升高而恶化；
- 失败请求数：正常测试应为 0，若大于 0，需检查服务端是否稳定、并发数是否过高、网络是否正常。
```

# 9.3 高级参数扫描与可视化

当需要批量测试不同参数组合（如不同 batch-size、tensor-parallel-size、request-rate）的性能差异时，可使用 vllm bench 的高级命令（sweep serve/sweep plot/sweep merge），实现参数扫描、结果可视化与合并，提升调优效率。

## 9.3.1 参数扫描概念与应用场景

### 9.3.1.1 概念

参数扫描（Parameter Sweep）是指批量设置不同的参数组合，自动执行基准测试，收集各组合的性能数据，核心目的是快速找到最优参数组合，定位参数对性能的影响规律。vLLM 官方通过 `vllm bench sweep serve` 命令实现参数扫描，支持服务端参数（如 tensor-parallel-size）与客户端测试参数（如 concurrency）的联合扫描。

核心目标，找到最适合的参数搭配。

### 9.3.1.2 应用场景

核心应用场景：

- 多参数调优：如同时测试不同 tensor-parallel-size（1/2/4）、concurrency（10/20/30）、max-num-batched-tokens（2048/4096/8192）的组合性能；

- 硬件适配：针对不同 GPU 型号，扫描最优参数组合，适配硬件资源；

- 场景适配：针对不同业务场景（低延迟/高吞吐量），扫描对应的最优参数，如实时场景优先降低延迟，批量场景优先提升吞吐量；

- 版本对比：扫描不同 vLLM 版本、不同模型版本的性能差异，验证版本迭代效果。

## 9.3.2 运行参数扫描 (vllm bench sweep serve)

该命令是参数扫描的核心命令，支持通过命令行参数或 JSON 配置文件，定义参数组合，自动启动服务、执行测试、保存结果。严格遵循官方文档语法与参数规范。

### 9.3.2.1 核心参数

#### 1. 扫描控制参数

- `--serve-cmd`：必填，vLLM 服务启动命令，如 `"vllm serve --model/mnt/moark-models/Qwen3-0.6B/ --port 8000"`；

- `--bench-cmd`：必填，在线服务基准测试命令（即 `vllm bench serve` 的基础命令），如 `"vllm bench serve --backend vllm --base-url http://localhost:8000/v1 --model/mnt/moark-models/Qwen3-0.6B/"`；

- `--output-dir`：必填，扫描结果输出目录，将自动生成子目录存储各参数组合的测试结果，默认目录为 `sweep_results`；

- `--num-trials`：可选，每个参数组合的测试次数，默认值为 1，建议取值 ≥ 3，消除偶然误差。

#### 2. 参数配置参数

- `--serve-params`：可选，服务端参数组合，格式为 `参数名=值1,值2;参数名=值1,值2`，如 `"tensor_parallel_size=1,2;max_num_batched_tokens=2048,4096"`；

- `--bench-params`：可选，客户端测试参数组合，格式与 `--serve-params` 一致，如 `"concurrency=10,20;num_prompts=100,200"`；

- `--params-file`：可选，参数配置 JSON 文件路径，当参数组合较多时，推荐使用该参数，替代 `--serve-params` 和`--bench-params`。

### 9.3.2.2 参数文件格式示例（JSON，官方标准格式）

```json
1: 准备server不同参数的json文件
(base) root@9c74b76a6bea:/data# cat serve_params.json 
[
    {
        "max-num-batched-tokens": 2048,
        "gpu-memory-utilization": 0.3
    },
    {
        "max-num-batched-tokens": 4096,
        "gpu-memory-utilization": 0.5
    },
    {
        "max-num-batched-tokens": 8192,
        "--gpu-memory-utilization": 0.7
    },
    {
        "max-num-batched-tokens": 16384,
        "pu-memory-utilization": 0.9
    }
]
2: 准备bench serve不通参数的文件
(base) root@9c74b76a6bea:/data# cat bench_params.json 
[
    {"max-concurrency": 10, "temperature": 0.7},
    {"max-concurrency": 30, "temperature": 0.3},
    {"max-concurrency": 60, "temperature": 1.0}
]
3: 开始测试
(base) root@9c74b76a6bea:/data# vllm bench sweep serve \
> --serve-cmd "vllm serve /mnt/moark-models/Qwen3-0.6B/ --host 0.0.0.0 --port 8000" \
> --bench-cmd "vllm bench serve --base-url http://localhost:8000 --model /mnt/moark-models/Qwen3-0.6B/ --num-prompts 150" \
> --serve-params serve_params.json \
> --bench-params bench_params.json \
> --output-dir test_dir
```



### 9.3.2.3 运行示例与输出目录结构

#### 运行示例：

```bash
(base) root@9c74b76a6bea:/data# vllm bench sweep serve \
> --serve-cmd "vllm serve /mnt/moark-models/Qwen3-0.6B/ --host 0.0.0.0 --port 8000" \
> --bench-cmd "vllm bench serve --base-url http://localhost:8000 --model /mnt/moark-models/Qwen3-0.6B/ --num-prompts 150" \
> --serve-params serve_params.json \
> --bench-params bench_params.json \
> --output-dir test_dir
```

#### 输出目录结构（官方标准结构）：

![image-20260308123245375](http://www.410166399.xyz/image-20260308123245375.png)

![image-20260308133907177](http://www.410166399.xyz/image-20260308133907177.png)

## 9.3.3 扫描结果可视化 (vllm bench sweep plot)

该命令用于将参数扫描的结果进行可视化，生成折线图、热力图等，直观展示参数与性能指标的关系，支持数据筛选、分箱、图表样式自定义。严格遵循官方文档语法与参数规范，依赖 matplotlib 库（需提前安装：`pip install matplotlib`）。

### 9.3.3.1 绘图参数

官方文档明确的核心参数（按功能分类）：

#### 1. 核心绘图参数

- `--input-dir`：必填，参数扫描结果的主输出目录（即 `vllm bench sweep serve` 的 `--output-dir`）；

- `--var-x`：必填，X 轴变量（即扫描的参数名），如 `concurrency`、`tensor_parallel_size`；

- `--var-y`：必填，Y 轴变量（即性能指标名），如 `throughput_tokens_per_sec`、`latency_distribution_ms.ttft.p99`；

- `--curve-by`：可选，分组变量，用于绘制多条曲线（不同分组对应不同曲线），如`tensor_parallel_size`；

- `--row-by`：可选，行分组变量，用于生成子图（按行分组），如 `max_num_batched_tokens`；

- `--col-by`：可选，列分组变量，用于生成子图（按列分组），如 `num_prompts`。

#### 2. 数据筛选与分箱参数

- `--filter-by`：可选，数据筛选条件，格式为 `参数名=值1,值2`，如 `"tensor_parallel_size=1,2;max_num_batched_tokens=4096"`，仅保留符合条件的数据；

- `--bin-by`：可选，分箱变量，用于将连续参数离散化，格式为 `参数名=区间数`，如 `"concurrency=3"`，将 concurrency 分为 3 个区间。

#### 3. 坐标轴与图表样式参数

- `--scale-x`：可选，X 轴缩放方式，支持 `linear`（线性，默认）、`log`（对数）；

- `--scale-y`：可选，Y 轴缩放方式，支持 `linear`、`log`；

- `--fig-name`：可选，图表文件名，默认值为 `sweep_plot.png`，支持 png、pdf、svg 格式；

- `--fig-size`：可选，图表尺寸，格式为 `宽,高`，默认值为 `10,6`（单位：英寸）；

- `--title`：可选，图表标题，默认值为 `Parameter Sweep Results`。

### 9.3.3.2 使用示例与图表解读

#### 使用示例：

```bash
假设你想分析不同并发数（concurrency）对请求吞吐量（request_throughput）的影响，并用不同颜色的曲线区分批处理 token 数（max-num-batched-tokens）和温度（temperature）

(base) root@9c74b76a6bea:/data# vllm bench sweep plot ./test_dir   --var-x concurrency   --var-y request_throughput   --curve-by max-num-batched-tokens   --row-by temperature

```

#### 图表解读（官方标准解读）：

生成的图表为多子图折线图，核心解读点：

- X 轴：并发数（concurrency），从 10 到 30 递增；

- Y 轴：吞吐量（tokens/s），数值越高，性能越好；

  
