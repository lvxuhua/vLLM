# 第六章：vLLM 架构概述（官方文档直译）

## 本章目标：

• 熟练掌握vLLM的架构

• 熟练掌握Entrypoint

• 熟练掌握LLM engine

• 熟练掌握AsyncLLM实例

• 熟练掌握调度器、执行器、model_runner

• 熟练掌握DP Coordinate

• 熟练掌握vllm serve 加载模型的启动过程

## 作业：

• 绘制vllm架构图保存在/data/image目录下

• 写一个文档包含vllm组件以及组件之间的交互。同时包含vllm加载模型的启动阶段保存在/data/result/vllm.txt

# vLLM总体架构图解：

![image-20260320125218071](http://www.410166399.xyz/image-20260320125218071.png)

# 架构概述

# 入口点

vLLM提供了多个用于与系统交互的入口点，下图展示了这些入口点之间的关联关系。

![image-20260320125410919](http://www.410166399.xyz/image-20260320125410919.png)

### 入口点对应关系

LLM 类 ↔ 兼容OpenAI的API服务器

LLMEngine 类 ↔ AsyncLLMEngine 类

## LLM 类

LLM类提供了用于执行离线推理的主要Python接口，离线推理即无需借助独立的模型推理服务器，直接与模型进行交互。

以下是`LLM`类的使用示例代码：

```python
from vllm import LLM, SamplingParams

# 定义输入提示词列表
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The largest ocean is",
]

# 定义采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 使用OPT-125M模型初始化LLM引擎
llm = LLM(model="facebook/opt-125m")

# 针对输入提示词生成输出结果
outputs = llm.generate(prompts, sampling_params)

# 打印生成的输出内容
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

更多API详情可参第二章离线推理。

LLM类的代码位于：`vllm/entrypoints/llm.py`

## 兼容OpenAI的API服务器

vLLM的第二个核心交互接口是其兼容OpenAI的API服务器，可通过`vllm serve`命令启动该服务器。

```bash
vllm serve <model>
```

vLLM命令行工具（CLI）的代码位于：`vllm/entrypoints/cli/main.py`

有时你可能会看到直接调用API服务器入口点的用法，而非通过vLLM CLI命令，示例如下：

```bash
python -m vllm.entrypoints.openai.api_server --model <model>
```

**警告**：`python -m vllm.entrypoints.openai.api_server` 该用法已被弃用，未来版本中可能不再支持。



# V1 版本进程架构

vLLM V1采用多进程架构，实现功能解耦并最大化推理吞吐量。理解该架构，对于部署时合理规划CPU资源配置至关重要。核心进程类型如下：

## API服务器进程

API服务器进程负责处理HTTP请求（例如兼容OpenAI的API请求），执行输入处理（分词、多模态数据加载），并将结果流式返回给客户端。该进程通过ZMQ套接字与引擎核心进程进行通信。

默认情况下，仅启动1个API服务器进程；启用数据并行时，API服务器进程数量会自动扩展，与数据并行规模保持一致。也可通过`--api-server-count`参数手动配置进程数量。每个API服务器进程通过ZMQ与所有引擎核心进程建立多对多拓扑连接，支持任意API服务器进程将请求路由至任意引擎核心进程。

每个API服务器进程会启用多个CPU线程处理媒体加载任务，线程数量由`VLLM_MEDIA_LOADING_THREAD_COUNT`控制，默认值为8。



## 引擎核心进程

引擎核心进程负责运行调度器、管理KV缓存，并协调各个GPU工作进程执行模型推理。该进程会运行一个忙循环，持续调度请求并将计算任务分发给GPU工作进程。

每个数据并行秩对应1个引擎核心进程，例如配置`--data-parallel-size 4`时，会启动4个引擎核心进程。



## GPU工作进程

每块GPU由一个专属的工作进程管理，该进程负责加载模型权重、执行模型前向传播、管理GPU显存。工作进程仅与归属的引擎核心进程进行通信。

每块GPU对应1个工作进程，单个引擎核心进程下的GPU工作进程总数 = 张量并行规模 × 流水线并行规模。

该部分代码位于：`vllm/v1/executor/multiproc_executor.py` 和 `vllm/v1/worker/gpu_worker.py`

## 数据并行协调进程（条件启动）

启用数据并行时（`--data-parallel-size > 1`），会额外启动一个协调进程，负责实现各数据并行秩之间的负载均衡，并协调混合专家（MoE）模型的同步前向传播操作。

仅启用数据并行时，会启动1个数据并行协调进程；未启用则不启动。

该部分代码位于：`vllm/v1/engine/coordinator.py`

## 进程数量汇总

假设部署环境配置为：N块GPU、TP（张量并行规模）、DP（数据并行规模）、A（API服务器进程数量），各进程数量如下表所示：

|进程类型|数量|说明|
|---|---|---|
|API服务器|A（默认等于DP）|处理HTTP请求与输入处理|
|引擎核心|DP（默认1）|调度器与KV缓存管理|
|GPU工作进程|N（等于DP×PP×TP）|每块GPU对应1个，执行模型前向传播|
|数据并行协调进程|DP>1时为1，否则为0|实现各数据并行秩之间的负载均衡|
|总计|A + DP + N（DP>1时额外加1）|-|
### 示例1：单节点4卡部署

启动命令：`vllm serve -tp=4`

进程组成：1个API服务器 + 1个引擎核心 + 4个GPU工作进程 = 总计6个进程

![image-20260320125903637](http://www.410166399.xyz/image-20260320125903637.png)

### 示例2：8卡数据并行部署

启动命令：`vllm serve -tp=2 -dp=4`

进程组成：4个API服务器 + 4个引擎核心 + 8个GPU工作进程 + 1个数据并行协调进程 = 总计17个进程

![image-20260320125938122](http://www.410166399.xyz/image-20260320125938122.png)



# LLM引擎

`LLMEngine`和`AsyncLLMEngine`类是vLLM系统的核心组件，负责处理模型推理与异步请求任务，核心交互链路如下：

LLM类 / 兼容OpenAI的API服务器 → LLMEngine / AsyncLLMEngine → 输入处理 → 调度 → 执行 → 模型处理 → 输出处理

![image-20260320130016008](http://www.410166399.xyz/image-20260320130016008.png)

## LLMEngine

`LLMEngine`类是vLLM引擎的核心，负责接收客户端请求并驱动模型生成输出结果。LLMEngine涵盖输入处理、模型执行（支持跨多主机、多GPU分布式部署）、请求调度、输出处理全流程。

- **输入处理**：通过指定的分词器对输入文本进行分词处理

- **调度**：确定每一步推理中需要处理的请求集合

- **模型执行**：管理语言模型的推理流程，包括跨多GPU的分布式执行

- **输出处理**：处理模型生成的结果，将模型输出的词元ID解码为人类可读的文本



## AsyncLLMEngine

`AsyncLLMEngine`类是LLMEngine类的异步封装类，基于`asyncio`构建后台循环，持续处理 incoming（传入）的请求。AsyncLLMEngine专为在线服务场景设计，可处理高并发请求，并将结果流式返回给客户端。

兼容OpenAI的API服务器基于AsyncLLMEngine实现，同时vLLM还提供了一个简化的演示版API服务器，代码位于：`vllm/entrypoints/api_server.py`

AsyncLLMEngine的代码位于：`vllm/engine/async_llm_engine.py`

# Worker（工作进程）

工作进程是执行模型推理的进程，vLLM遵循行业通用规范：一个进程管控一个加速设备（例如GPU）。例如，张量并行规模设为2、流水线并行规模设为2时，总共会启动4个工作进程。

工作进程通过`rank`（全局秩）和`local_rank`（本地秩）进行标识：`rank`用于全局编排调度，`local_rank`主要用于分配加速设备，以及访问文件系统、共享内存等本地资源。

# Model Runner（模型运行器）

每个工作进程都包含一个模型运行器对象，负责加载并运行模型。模型的核心执行逻辑均封装在此处，例如输入张量预处理、CUDA图捕获等操作。

# Model（模型）

每个模型运行器对象对应一个模型对象，即实际的`torch.nn.Module`实例。各类配置对最终加载的模型类的影响，可参考Hugging Face集成相关文档。

# 类层级结构

![image-20260320130129639](http://www.410166399.xyz/image-20260320130129639.png)

该类层级结构背后，有几项重要的设计考量：

## 1. 可扩展性

层级中的所有类都会接收一个包含全部所需信息的配置对象，`VllmConfig`是核心配置对象，会在各组件间传递。vLLM的类层级较深，每个类仅需读取自身所需的配置项，将所有配置封装在一个对象中，可便捷完成配置传递与访问。

大模型推理领域迭代速度极快，若需新增功能，假设该功能仅涉及模型运行器，只需在`VllmConfig`类中新增对应配置项即可；由于配置对象全局传递，无需修改引擎、工作进程、模型类的构造函数，模型运行器可直接读取新增配置，大幅降低开发成本。

## 2. 统一性

模型运行器需要统一的接口来创建和初始化模型，vLLM支持50余种主流开源模型，各类模型的初始化逻辑各不相同。若模型构造函数签名不统一，模型运行器需要编写复杂且易出错的校验逻辑，才能适配不同模型的初始化方式。

通过统一模型类的构造函数签名，模型运行器无需感知具体模型类型，即可快速完成模型的创建与初始化，该设计也便于模型组合。例如多模态大模型通常包含视觉模型和语言模型，统一构造接口后，可轻松创建两类子模型并组合为完整的多模态模型。

**注意**：为适配该设计，所有vLLM模型的构造函数签名已更新为：

构造函数采用仅限关键字参数的形式，避免参数传递错误，若使用旧版配置传参，构造函数会直接抛出异常。vLLM官方已完成所有内置模型的签名更新，对于外部注册的模型，开发者需自行适配，可通过兼容代码适配新旧版本vLLM，示例如下：

通过该方式，模型可同时兼容新旧版本的vLLM。

```
`def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):` `class MyOldModel(nn.Module):
    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
    ) -> None:
        ...

from vllm.config import VllmConfig
class MyNewModel(MyOldModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        super().__init__(config, cache_config, quant_config, lora_config, prefix)

from packaging import version
if version.parse(__version__) >= version.parse("0.6.4"):
    MyModel = MyNewModel
else:
    MyModel = MyOldModel`

## 
```

## 3. 初始化阶段完成权重分片与量化

张量并行、模型量化等功能需要修改模型权重，实现方式分为两种：一是模型初始化后修改权重，二是模型初始化过程中修改权重，vLLM选用第二种方式。

第一种方式对大模型扩展性极差，例如使用16块80GB显存的H100 GPU运行405B模型（权重约810GB），理想状态下每块GPU仅需加载50GB权重；若在模型初始化后修改权重，每块GPU都需要先加载完整的810GB权重，再进行分片，会产生巨大的显存开销。

而在初始化阶段修改权重，模型的每一层仅会创建自身所需的权重分片，显存开销大幅降低，量化操作也遵循该设计逻辑。此外，模型构造函数新增了`prefix`参数，支持模型根据前缀差异化初始化，适配非均匀量化场景（模型不同部分采用不同量化策略）。`prefix`对于顶层模型默认为空字符串，子模型（如视觉、语言子模型）则为对应名称，通常与检查点文件中的模块状态字典名称一致。
