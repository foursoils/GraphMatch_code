以下是针对**基于图匹配网络（GMN）的多模态双重注入大语言模型（LLM）幻觉检测方案**的完整架构与技术说明文段，可直接用于论文的架构设计（Methodology）章节或技术报告中：

## 基于GMN双重注入的LLM事实核查与纠偏架构说明

本方案提出了一种新颖的 **宏观-微观协同纠偏机制（Macro-Micro Collaborative Correction）** ，旨在利用图匹配网络（GMN）提取的高维结构化拓扑特征，动态指导大语言模型（LLM）的思维链（CoT）推理并抑制幻觉生成。方案整体采用双路异构特征融合设计，将待检测的断言图（**$G_{\text{claim}}$**）与参考文档图（**$G_{\text{doc}}$**）输入预训练的GMN中。在流经LLM第16层及以后的解码计算时，系统分别从“图级全局拓扑冲突”与“节点级细粒度实体冲突”两个维度，通过不同的数学通路对大模型进行“双重注入”微调。

### 1. 通路一：基于图级差异向量的“全局自注意力偏置”注入（Macro-Level）

图级注入旨在从宏观维度让大模型感知当前生成的文本推理与参考知识库之间的整体事实偏离度。

GMN最终聚合输出两个全图级向量：**$h_{\text{claim}}, h_{\text{doc}} \in \mathbb{R}^{B \times D_{\text{gmn}}}$**。系统首先计算两者的拓扑差值以捕获全局不匹配信号：

$$
\Delta h_G = h_{\text{claim}} - h_{\text{doc}}
$$

随后，通过一个可学习的线性映射层 **$W_{\text{graph}} \in \mathbb{R}^{D_{\text{gmn}} \times H_{\text{heads}}}$** 将该差值向量投影至LLM的多头注意力特征空间，生成全局图级偏置项：

$$
b_{\text{graph}} = \Delta h_G \cdot W_{\text{graph}}
$$

在LLM自注意力机制（Self-Attention）计算各Token间的关联得分矩阵 **$A = \frac{Q K^T}{\sqrt{d}} \in \mathbb{R}^{B \times H_{\text{heads}} \times L \times L}$** 时，将 **$b_{\text{graph}}$** 动态广播并作为背景偏置（Attention Bias）直接融入公式中：

$$
A_{\text{new}} = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d}} + \gamma \cdot \text{Unsqueeze}(b_{\text{graph}}) \right)
$$

其中 **$\gamma$** 为可学习的缩放标量。当两图结构冲突剧烈时，该偏置项将压制LLM自注意力层盲目的序列置信度，促使模型将计算权重向跨模态知识检索通路倾斜。

### 2. 通路二：基于节点级矩阵的“门控交叉注意力”注入（Micro-Level）

节点级注入旨在从微观维度让大模型在生成具体的实体、概念或逻辑关系时，实现字词级别的“事实查表与对齐”。

系统将GMN在传播层结束时保留的断言图与文档图的所有节点隐状态横向拼接，构建包含动态数量 **$N$** 个节点的局部知识矩阵 **$H_{\text{nodes}} \in \mathbb{R}^{B \times N \times D_{\text{gmn}}}$**。在LLM原有的Self-Attention模块之后、FFN（前馈网络）模块之前，强行嵌入一个全新的 **门控交叉注意力层（Gated Cross-Attention Layer）** 。

首先，利用全新初始化的线性层将变长的节点特征投影为LLM的键（Key）和值（Value）矩阵：

$$
K_g = H_{\text{nodes}} \cdot W_K, \quad V_g = H_{\text{nodes}} \cdot W_V
$$

此时，以经历完自注意力校准的LLM隐藏状态 **$H_{\text{llm}} \in \mathbb{R}^{B \times L \times D_{\text{llm}}}$** 作为查询矩阵（Query）：

$$
Q_l = H_{\text{llm}} \cdot W_Q
$$

通过缩放点积跨注意力算子计算文本Token对结构化实体的检索关联，并在其矩阵乘法中天然消解了动态节点数 **$N$** 的异构限制，输出符合LLM维度的上下文矩阵 **$\text{Context} \in \mathbb{R}^{B \times L \times D_{\text{llm}}}$**：

$$
\text{Context} = \text{Softmax}\left( \frac{Q_l \cdot K_g^T}{\sqrt{D_{\text{llm}}}} \right) \cdot V_g
$$

为了确保模型训练初期的稳定性，防止新特征破坏LLM原有的语言表征，本架构引入零初始化的门控隐式机制（Tanh Gating）进行残差融合：

$$
H_{\text{out}} = H_{\text{llm}} + \tanh(\alpha) \cdot \text{Context}
$$

其中门控标量 **$\alpha$** 显式初始化为0。随着微调训练的推进，**$\alpha$** 会自适应放大，使得LLM在思维链（CoT）生成具体的断言实体时，能精准通过Cross-Attention捕获事实冲突并实时扭转Token的概率分布，从而在根源上阻断幻觉的滋生。

### 3. 训练与微调策略

本架构在训练时采用高效的参数解耦微调方案（Parameter-Efficient Fine-Tuning）。大语言模型（LLM）的1至15层参数完全冻结（Freeze），以完整保留其深厚的通用语言理解功底。从第16层至最终的输出层，对LLM原有的自注意力投影矩阵与FFN全连接矩阵挂载 **LoRA（低秩适应）旁路** 。训练时，GMN匹配网络、新插入的Gated Cross-Attention层以及LLM后16层的LoRA参数进行端到端的联合训练。利用包含思维链分析的核查数据集，模型通过预测最终的事实分析轨迹与判别标签计算交叉熵损失（Cross-Entropy Loss），倒推梯度协同更新，最终使整个系统具备“深思熟虑、据实纠偏”的强健幻觉判别能力。
