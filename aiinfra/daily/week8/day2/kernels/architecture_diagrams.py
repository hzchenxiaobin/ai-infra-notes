# architecture_diagrams.py —— 生成 Mini 引擎 Mermaid 架构图集
# 运行命令: python architecture_diagrams.py
# 依赖: 仅标准库
# 作用: 输出 4 张 Mermaid 图（架构/时序/时间线/状态机）的 Markdown 代码


def system_architecture():
    return """```mermaid
graph TD
    User[User Request] --> Queue[ThreadSafeRequestQueue<br/>条件变量+优先级]
    Queue --> Scheduler[Scheduler<br/>Continuous Batching]
    Scheduler --> Worker[Worker<br/>Model Forward]
    Worker --> KV[KV Cache Manager]
    Worker --> Sampler[Sampler]
    Sampler --> Result[Result Queue]
    Result --> User
    subgraph CUDA Kernels
        GEMM[GEMM 72%]
        FA[FlashAttention 2.1x]
        SM[Softmax]
        LN[RMSNorm]
    end
    Worker --> GEMM & FA & SM & LN
```"""


def request_sequence():
    return """```mermaid
sequenceDiagram
    participant U as User
    participant E as Engine
    participant Q as Queue
    participant S as Scheduler
    participant W as Worker
    participant K as KV Cache
    U->>E: submit(prompt)
    E->>Q: enqueue (WAITING)
    S->>Q: get_batch()
    S->>W: schedule (RUNNING)
    W->>K: read/write KV
    W->>W: forward + sample
    W->>S: outputs
    S->>U: set_result (FINISHED)
```"""


def state_machine():
    return """```mermaid
stateDiagram-v2
    [*] --> WAITING: submit()
    WAITING --> RUNNING: schedule()
    RUNNING --> FINISHED: 完成
    RUNNING --> TIMEOUT: 超时
    RUNNING --> CANCELLED: cancel()
    WAITING --> CANCELLED: cancel()
    FINISHED --> [*]
    TIMEOUT --> [*]
    CANCELLED --> [*]
```"""


def batching_timeline():
    return """| 请求 | iter0 | iter1 | iter2 | iter3 | iter4 |
|------|-------|-------|-------|-------|-------|
| R1   | prefill | decode | decode | done ✓ | |
| R2   | prefill | decode | decode | decode | done ✓ |
| R3   | | | prefill | decode | decode |
| **batch** | **2** | **2** | **3** | **2** | **1** |"""


DIAGRAMS = {
    "系统架构图": system_architecture,
    "请求时序图": request_sequence,
    "状态机图": state_machine,
    "Batching 时间线": batching_timeline,
}

if __name__ == "__main__":
    for name, fn in DIAGRAMS.items():
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}\n")
        print(fn())
