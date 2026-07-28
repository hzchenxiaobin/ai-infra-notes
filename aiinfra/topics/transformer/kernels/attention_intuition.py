# attention_intuition.py —— Day 1: 字符级 tokenization、embedding 语义与注意力直觉
# 运行：python3 attention_intuition.py
# 仅依赖 PyTorch，CPU 即可运行
import torch
import torch.nn.functional as F


def demo_tokenize_and_embed():
    """字符级 tokenization + embedding lookup：从文本到向量。"""
    print("=" * 64)
    print("实验 1：字符级 tokenization → embedding lookup")
    print("=" * 64)
    text = "hello"
    chars = sorted(set(text))                     # 建字符表
    stoi = {c: i for i, c in enumerate(chars)}    # char -> id
    itos = {i: c for c, i in stoi.items()}        # id -> char
    ids = torch.tensor([stoi[c] for c in text])   # tokenize 成 id 序列
    print(f"文本: {text!r}")
    print(f"字符表: {chars}  →  ids: {ids.tolist()}")

    # embedding lookup：把 id 映射成稠密向量（权重可学习）
    torch.manual_seed(0)
    embed = torch.nn.Embedding(num_embeddings=len(chars), embedding_dim=4)
    vectors = embed(ids)                          # (T, 4)
    print(f"embedding 后形状: {tuple(vectors.shape)}  (T=5, d=4)")
    print(f"'l' 的向量: {[round(x, 4) for x in vectors[2].tolist()]}")

    # 关键：embedding lookup 在数学上等价于 one-hot @ 权重矩阵
    onehot = F.one_hot(ids, num_classes=len(chars)).float()
    lookup_via_matmul = onehot @ embed.weight     # 与 embed(ids) 完全一致
    print(f"one-hot @ E 是否等价于 embedding lookup: "
          f"{torch.allclose(lookup_via_matmul, vectors, atol=1e-6)}")

    # one-hot 的致命缺陷：任意两个不同字符正交，余弦相似度恒为 0
    cos = F.cosine_similarity(onehot[0], onehot[1], dim=0)
    print(f"one-hot cos('h','e') = {cos:.3f}  ← 所有不同字符都正交，无语义信息")


def demo_embedding_semantics():
    """手构造 embedding 观察余弦相似度与词向量类比。"""
    print("\n" + "=" * 64)
    print("实验 2：embedding 的语义空间（余弦相似度 + 类比）")
    print("=" * 64)
    vocab = ["king", "queen", "man", "woman", "boy", "girl"]
    # 4 个维度分别对应：[王权, 男性, 女性, 成年]（真实模型维度不可解释，这里为示意）
    embed = torch.tensor([
        [1, 1, 0, 1],  # king
        [1, 0, 1, 1],  # queen
        [0, 1, 0, 1],  # man
        [0, 0, 1, 1],  # woman
        [0, 1, 0, 0],  # boy
        [0, 0, 1, 0],  # girl
    ], dtype=torch.float32)

    cos = F.cosine_similarity(embed.unsqueeze(1), embed.unsqueeze(0), dim=-1)
    print("余弦相似度矩阵：")
    print("        " + "  ".join(f"{w:>5}" for w in vocab))
    for i, w in enumerate(vocab):
        print(f"{w:>5}  " + "  ".join(f"{cos[i, j]:.2f}" for j in range(len(vocab))))
    print(f"\ncos(king, man)  = {cos[0, 2]:.3f}  (同为男性/成年 → 高)")
    print(f"cos(man, boy)   = {cos[2, 4]:.3f}  (同为男性 → 高)")
    print(f"cos(king, girl) = {cos[0, 5]:.3f}  (几乎无关 → 低)")

    # 经典词向量类比：king - man + woman ≈ queen
    analogy = embed[0] - embed[2] + embed[3]
    sims = F.cosine_similarity(analogy, embed, dim=-1)
    best = sims.argmax().item()
    print(f"\n类比 king - man + woman → 最接近 '{vocab[best]}' (cos={sims[best]:.3f})")
    print("→ embedding 空间的'方向'承载了语义关系，这是 one-hot 做不到的")


def demo_toy_attention():
    """3 行 PyTorch 算 toy 注意力权重矩阵并画文本热力图。"""
    print("\n" + "=" * 64)
    print("实验 3：toy 注意力权重矩阵（自注意力，Q=K=V=embedding）")
    print("=" * 64)
    vocab = ["king", "queen", "man", "woman", "boy", "girl"]
    embed = torch.tensor([
        [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1],
        [0, 0, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0],
    ], dtype=torch.float32)

    # ---- 注意力权重矩阵，核心就 3 行（今天先不缩放、不掩码，只看直觉）----
    scores = embed @ embed.T                  # 1) QK^T：两两相似度
    attn = F.softmax(scores, dim=-1)          # 2) softmax 归一化成权重
    out = attn @ embed                        # 3) 加权求和 V（每个词的新表示）

    print("注意力权重矩阵（行=查询词，列=被看词），每行和为 1：")
    print("        " + "  ".join(f"{w:>5}" for w in vocab))
    for i, w in enumerate(vocab):
        print(f"{w:>5}  " + "  ".join(f"{attn[i, j]:.2f}" for j in range(len(vocab))))
    print(f"\nking → queen/man 权重 ({attn[0, 1]:.2f}/{attn[0, 2]:.2f}) "
          f"远大于 king → girl ({attn[0, 5]:.2f})")
    print("→ 每个词'多看语义相关的词'，这就是注意力 = 软检索")

    print("\n文本热力图（█ 越多 = 权重越高）：")
    print("        " + "  ".join(f"{w:>5}" for w in vocab))
    for i, w in enumerate(vocab):
        bars = "  ".join("█" * int(round(attn[i, j].item() * 20)) for j in range(len(vocab)))
        print(f"{w:>5}  {bars}")

    print(f"\n加权求和后 king 的新表示: {[round(x, 4) for x in out[0].tolist()]}")
    print("⚠️ 这份权重完全基于语义相似度，对'顺序'毫无感知——")
    print("   打乱词序，每行权重分布不变。这正是 Day 3 要引入位置编码的原因。")


if __name__ == "__main__":
    demo_tokenize_and_embed()
    demo_embedding_semantics()
    demo_toy_attention()
