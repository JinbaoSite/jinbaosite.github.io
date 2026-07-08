# 混合精度训练与分布式训练（DDP）

当模型规模达到数十亿参数、数据达到 TB 级别，单卡训练已经不可能。**混合精度训练（Mixed Precision / AMP）**用半精度（FP16 / BF16）节省显存和加速；**分布式数据并行（DDP）**把数据切片到多张卡并行训练。本文系统讲解这两大技术的原理、PyTorch 实现与踩坑指南。

---

## 一、为什么需要混合精度？

### 1.1 浮点格式回顾

| 格式     | 符号 | 指数 | 尾数 | 字节数 | 数值范围       | 精度          |
| -------- | ---- | ---- | ---- | ------ | -------------- | ------------- |
| FP32     | 1    | 8    | 23   | 4      | ±3.4×10³⁸      | 高（~7 位有效数字） |
| FP16     | 1    | 5    | 10   | 2      | ±65504         | 中（~3 位有效数字） |
| BF16     | 1    | 8    | 7    | 2      | ±3.4×10³⁸      | 低（~2 位有效数字） |
| TF32     | 1    | 8    | 10   | 4      | ±3.4×10³⁸      | 中（GPU 默认张量核） |

### 1.2 FP16 的优势

- **显存减半**：模型参数、激活、梯度都从 4 字节降到 2 字节 → 同样的卡能跑 2 倍大的模型 / 2 倍大的 batch。
- **计算加速**：Volta 及之后 GPU 的 **Tensor Core** 对 FP16 矩阵乘有专门硬件加速，理论 2×–8× 加速。
- **带宽减半**：PCIe、NVLink 传输同样数据量耗时减半。

### 1.3 FP16 的问题

- **数值范围窄**：最大值仅 65504，**大梯度容易溢出（inf/nan）**。
- **精度损失**：~3 位有效数字，**小梯度可能下溢（变为 0）**。
- **更新不一致**：参数用 FP16 存储，累加更新时误差累积。

### 1.4 BF16 的折中

BF16 用 8 位指数，**数值范围和 FP32 一样大**，但尾数只有 7 位（精度低）。**现代大模型训练几乎都用 BF16**（A100、H100 原生支持），无需 loss scaling。

---

## 二、混合精度训练核心思想

**保留 FP32 主副本**：参数和优化器状态保存在 FP32；前向/反向用 FP16；最后更新时把梯度 cast 回 FP32 更新 FP32 主副本，再 cast 为 FP16 用于下一轮前向。

```
FP32 master weights ──cast──> FP16 weights ──> forward ──> FP16 loss
                                                       │
FP32 master weights <──cast── FP16 weights <──update── FP32 updated weights
                          │                
                          └──> backward (FP16)  
```

### 2.1 Loss Scaling

针对 FP16 数值范围窄的问题，**在 loss 上乘一个大数 S（如 65536）**，让所有梯度同步放大，反向时再除回来。**Loss Scaling 是 FP16 训练不可或缺的一步**。

### 2.2 三种精度的角色

| 组件       | FP16 | FP32 | BF16 |
| ---------- | ---- | ---- | ---- |
| 模型参数   | ✓（存储） | ✓（主副本） | ✓（直接用） |
| 激活       | ✓     |      | ✓    |
| 梯度       | ✓（反向时） | ✓（累加） | ✓  |
| 优化器状态 |      | ✓    | ✓    |
| Loss       | ✓     |      | ✓    |

---

## 三、PyTorch AMP 实战

### 3.1 FP16 混合精度（需要 loss scaling）

```python
import torch
from torch.cuda.amp import GradScaler, autocast

model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = GradScaler()  # 自动 loss scaling

for x, y in dataloader:
    x, y = x.cuda(), y.cuda()
    optimizer.zero_grad()

    # autocast 上下文内的运算用 FP16
    with autocast():
        pred = model(x)
        loss = criterion(pred, y)

    # loss scaling + 反向
    scaler.scale(loss).backward()

    # 更新前 unscale 梯度,可选地做梯度裁剪
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 优化器更新;scaler 会自动调整 scale factor
    scaler.step(optimizer)
    scaler.update()
```

`GradScaler` 会动态调整 scale factor：

- 如果连续若干 step 没出现 inf/nan，scale × 2（让梯度用更大值）
- 如果出现 inf/nan，scale ÷ 2 并跳过该 step

### 3.2 BF16 混合精度（更简单）

```python
from torch.cuda.amp import autocast

for x, y in dataloader:
    x, y = x.cuda(), y.cuda()
    optimizer.zero_grad()

    # dtype=bfloat16,无需 scaler
    with autocast(dtype=torch.bfloat16):
        pred = model(x)
        loss = criterion(pred, y)

    loss.backward()
    optimizer.step()
```

BF16 训练**不需要 GradScaler**，代码更简洁。**这是 A100/H100 上的推荐做法**。

### 3.3 完整训练脚本骨架

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, optimizer, criterion, device, use_bf16=True):
    model.train()
    scaler = None if use_bf16 else GradScaler()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with autocast(dtype=amp_dtype):
            pred = model(x)
            loss = criterion(pred, y)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)
```

---

## 四、分布式数据并行（DDP）

### 4.1 DDP 的工作原理

DDP（DistributedDataParallel）在每张 GPU 上**复制一份完整模型**，把一个 global batch 平均切分到各卡：

```
Global Batch = 256, 4 GPUs
  ├── GPU 0: data[0:64],  复制模型
  ├── GPU 1: data[64:128]
  ├── GPU 2: data[128:192]
  └── GPU 3: data[192:256]

每个 GPU 独立 forward + backward → NCCL all-reduce 同步梯度 → 各卡独立更新
```

### 4.2 DDP vs DP vs FSDP

| 特性          | DataParallel (DP) | DistributedDataParallel (DDP) | FSDP (Fully Sharded DP) |
| ------------- | ----------------- | ------------------------------ | ----------------------- |
| 通信后端      | Gloo              | NCCL / Gloo                    | NCCL                   |
| 多机支持      | ✗（单进程多线程） | ✓                              | ✓                       |
| 性能          | 慢（GIL + 单进程瓶颈） | **快**                       | 最快（大模型必需） |
| 模型分片      | ✗                 | ✗（每卡全模型）                | ✓（参数/优化器/梯度都分片） |
| 显存占用      | 全模型 × 2 卡数    | 全模型 × 卡数                  | 全模型                  |
| 适用规模      | < 1B 参数         | 1B–13B 参数                    | > 13B 参数              |

### 4.3 DDP 启动方式

**方式 1：`torchrun`（推荐）**

```bash
# 单机 4 卡
torchrun --nproc_per_node=4 train.py

# 多机:每台机器单独启动
# Master node:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr="master_ip" --master_port=12345 train.py
# Worker node:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr="master_ip" --master_port=12345 train.py
```

**方式 2：环境变量手动**

```bash
RANK=0 WORLD_SIZE=4 MASTER_ADDR=localhost MASTER_PORT=12345 \
    python train.py
# 每卡单独启动一次,设置不同的 RANK
```

### 4.4 DDP 训练代码模板

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup_ddp():
    """初始化分布式进程组"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def main():
    rank = setup_ddp()
    is_main = (rank == 0)

    # 1. 模型放到当前 GPU
    model = MyModel().cuda(rank)
    model = DDP(model, device_ids=[rank])

    # 2. 数据集 + DistributedSampler
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler,
                        num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 关键:每个 epoch 重置 shuffle
        for x, y in loader:
            x, y = x.cuda(rank), y.cuda(rank)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(x)
                loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            if is_main and step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
```

### 4.5 DistributedSampler 的关键

```python
sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=64, sampler=sampler)
```

- 一定要用 `DistributedSampler`，不能用普通 `shuffle=True`（数据分布会重叠）
- 每个 epoch 开始时 `sampler.set_epoch(epoch)`，让不同 epoch 的 shuffle 不同

### 4.6 保存与加载 checkpoint

```python
# 只有 rank 0 保存
if dist.get_rank() == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),  # 注意 .module
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'checkpoint.pt')

# 所有 rank 加载(各加载到自己的模型)
map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
ckpt = torch.load('checkpoint.pt', map_location=map_location)
model.module.load_state_dict(ckpt['model_state_dict'])
```

### 4.7 通信原语

| 原语             | 含义 | DDP 中的用途 |
| ---------------- | ---- | ------------ |
| `all-reduce`     | 所有 rank 求和/平均，结果广播到所有 rank | **DDP 梯度同步** |
| `broadcast`      | 一 rank 广播到所有 rank              | 初始化参数       |
| `all-gather`     | 每 rank 提供数据，汇总到所有 rank     | FSDP 参数收集   |
| `reduce-scatter` | 每 rank 获得汇总结果的一部分         | FSDP 反向       |

DDP 在每次 backward 结束时自动插入 `all-reduce` 同步梯度。

### 4.8 性能优化

#### （1）Gradient Accumulation（梯度累积）

显存不够大 batch 时，用梯度累积模拟：

```python
accumulation_steps = 4
for i, (x, y) in enumerate(loader):
    with autocast(dtype=torch.bfloat16):
        loss = criterion(model(x.cuda()), y.cuda()) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### （2）Gradient Compression / Compression-aware 优化

字节通信压缩梯度（PowerSGD、1-bit Adam）。

#### （3）Overlap Communication 与 Computation

DDP 默认让 backward 梯度计算和 all-reduce **重叠**，减少通信延迟。

#### （4）大 batch 训练 + LAMB/LARS

参考 [optimizers.md](optimizers.md) 中 LAMB 的实现。

---

## 五、混合精度 + DDP 综合实践

### 5.1 多机多卡启动脚本

```bash
#!/bin/bash
# train_ddp.sh

NNODES=2          # 节点数
GPUS_PER_NODE=4   # 每节点 GPU 数
MASTER_ADDR="192.168.1.10"
MASTER_PORT=12345

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py
```

### 5.2 监控显存与吞吐

```python
# GPU 显存
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached:    {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# 吞吐量
torch.cuda.synchronize()
start = time.time()
# ... train ...
torch.cuda.synchronize()
print(f"Throughput: {num_samples / (time.time() - start):.1f} samples/s")
```

---

## 六、常见踩坑

| 问题 | 原因 | 解决 |
| ---- | ---- | ---- |
| Loss = NaN（FP16） | 梯度溢出 | 加 GradScaler、减小 LR、用 BF16 |
| 训练很慢（多卡比单卡还慢） | 通信瓶颈 / batch 太小 | 检查 NCCL，用大 batch |
| `RuntimeError: NCCL error` | 端口冲突 / 防火墙 | 改 MASTER_PORT，检查防火墙 |
| 多卡 loss 不下降 | DistributedSampler 没 set_epoch | 加上 `sampler.set_epoch(epoch)` |
| 加载 checkpoint 报错 | 直接 `model.state_dict()` 而不是 `model.module` | 加 `.module` |
| 多机训练 hang | 没设 NCCL_SOCKET_IFNAME | 设置环境变量指定网卡 |

---

## 七、一句话总结

> **BF16 + DDP + AdamW + GradClip + Warmup + Cosine** 是现代大模型训练的「**六件套**」。混合精度把单卡显存减半、速度翻倍；DDP 让训练规模线性扩展到多机多卡。两者组合是 LLM 时代的基础设施。