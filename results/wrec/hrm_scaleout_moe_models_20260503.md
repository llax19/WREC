# WREC Scale-Out 附录

## 目的

这一附录的作用是扩展 WREC 的问题动机，而不是改变主实验叙事。它不试图替代基于 Mixtral-8x7B trace 的主结果，只是展示：当 MoE 模型规模变化时，同样的 transfer-bound 压力会如何变化。

## 设定

所有 scale-out 对比都使用同一组简化 HRM screening 假设：

- dtype: `bf16`
- CPU-GPU bandwidth: measured `41.37220609315469 GB/s`
- KV reservation: `8 GB`
- active decode tokens: `32`
- requested resident fraction: `1.0`

参与比较的模型为：

- Mixtral-8x7B：本地官方 config
- Mixtral-8x22B：官方 Hugging Face config
- DeepSeek-MoE-16B：官方 Hugging Face config
- DBRX：公开 converted config proxy（`alpindale/dbrx-instruct`），因为当前环境中无法直接获取官方 Databricks Hugging Face config

## 汇总表

| model | source | total model GiB | expert GiB | GPU budget GiB | max feasible resident fraction | transfer ratio at max resident | all-resident feasible |
|---|---|---:|---:|---:|---:|---:|---|
| Mixtral-8x7B | local | 86.99 | 84.00 | 24 | 12.5% | 99.9% | False |
| Mixtral-8x7B | local | 86.99 | 84.00 | 48 | 37.5% | 99.8% | False |
| Mixtral-8x7B | local | 86.99 | 84.00 | 96 | 100.0% | 0.0% | True |
| Mixtral-8x22B | official HF | 261.94 | 252.00 | 24 | 0.0% | 99.9% | False |
| Mixtral-8x22B | official HF | 261.94 | 252.00 | 48 | 0.0% | 99.9% | False |
| Mixtral-8x22B | official HF | 261.94 | 252.00 | 96 | 25.0% | 99.9% | False |
| DBRX | public config proxy | 245.12 | 236.25 | 24 | 0.0% | 99.9% | False |
| DBRX | public config proxy | 245.12 | 236.25 | 48 | 12.5% | 99.9% | False |
| DBRX | public config proxy | 245.12 | 236.25 | 96 | 31.2% | 99.9% | False |
| DeepSeek-MoE-16B | official HF | 31.44 | 28.88 | 24 | 45.3% | 99.9% | False |
| DeepSeek-MoE-16B | official HF | 31.44 | 28.88 | 48 | 100.0% | 0.0% | True |
| DeepSeek-MoE-16B | official HF | 31.44 | 28.88 | 96 | 100.0% | 0.0% | True |

## 如何解读

这张表最重要的不是绝对参数量，而是四个模型在相同预算下的相对位置。

第一，DeepSeek-MoE-16B 足够小，在当前假设下到 `48 GB` 就可以 all-resident，因此它不是 WREC 目标场景中最强的例子。第二，Mixtral-8x22B 和 DBRX 即使在 `96 GB` 下也仍然高度 transfer-bound，这说明更大的 MoE 模型只会强化在线 expert cache scheduling 的动机。第三，Mixtral-8x7B 处在两者之间：它不会小到让问题自然消失，也不会大到让所有配置都显然 transfer-dominated，因此它是一个适合作为主实验模型的中间点。

## 范围与注意事项

这一附录刻意采用 config-only 的结构分析。它不使用这些更大模型的真实 route traces，因此不能被解读为“WREC 已经在这些模型上完成验证”。它唯一想表达的是：主文在 Mixtral-8x7B 上识别出的 transfer-bound regime，在更大 MoE 模型上至少同样相关，很多时候甚至更严重。
