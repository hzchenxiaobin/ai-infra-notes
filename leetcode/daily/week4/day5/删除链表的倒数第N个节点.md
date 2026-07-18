# 删除链表的倒数第N个节点

- **题目名称**：删除链表的倒数第N个节点
- **链接**：[19. 删除链表的倒数第N个节点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)
- **难度**：中等
- **标签**：链表、双指针

## 1. 题目概述

给定一个链表的头节点 `head`，删除链表的**倒数第** `n` **个节点**，并返回链表的头节点。要求实现一次遍历完成。

**示例 1**：

```text
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
解释：删除倒数第 2 个节点（值为 4 的节点），返回链表 1 → 2 → 3 → 5。
```

**示例 2**：

```text
输入：head = [1], n = 1
输出：[]
解释：链表只有一个节点，删除后为空。
```

**示例 3**：

```text
输入：head = [1,2], n = 1
输出：[1]
解释：删除倒数第 1 个节点（值为 2 的节点），返回链表 1。
```

**约束条件**：

- 链表中节点数目为 `sz`
- `1 <= sz <= 30`
- `0 <= Node.val <= 100`
- `1 <= n <= sz`
- **进阶**：你能尝试使用一趟扫描实现吗？

> 💡 这是链表题的经典招牌题。它引出两个通用技巧：**哑节点（dummy node）**——统一处理删除头节点的边界；**快慢双指针制造间距**——让快指针先走 `n` 步，制造 `n` 的间距，再同速推进，慢指针自然停在待删节点的前驱。这是"用先发优势锁定距离"的双指针模板，与 [环形链表](../../week3/day4/环形链表.md) 的"不同速追及"是双指针的两种典型用法。

---

## 2. 解题思路

### 2.1 暴力思路：两趟扫描（先数长度再找位置）

第一趟遍历求出链表长度 `L`，则倒数第 `n` 个节点即正数第 `L - n + 1` 个节点（下标从 1 开始）。第二趟走到第 `L - n` 个节点（待删节点的前驱），执行删除。

```text
# 第一趟：数长度
L = 0; cur = head
while cur: L++; cur = cur.next
# 第二趟：走到前驱（第 L-n 个）
target = L - n            # 前驱的下标（从 1 开始）
cur = head
for _ in range(target - 1): cur = cur.next
cur.next = cur.next.next  # 删除
```

时间 `O(n)`，空间 `O(1)`。能过，但需要**两趟扫描**，不满足进阶的"一趟扫描"要求。

> ⚠️ 暴力法的痛点：必须先知道总长度 `L` 才能算出"倒数第 n"对应"正数第几"。能否在一次扫描中就锁定前驱？关键在于**让一个指针提前出发**，提前量恰好等于 `n`，这样两个指针同时到达终点时，慢指针的位置就是前驱。

### 2.2 核心观察：快慢双指针制造 n 步间距

**两个关键技巧**：

1. **哑节点（dummy node）**：在 `head` 前加一个虚拟节点 `dummy`，`dummy.next = head`。这样删除头节点（`n == sz`）时，慢指针停在 `dummy`，`dummy.next = head` 的删除操作与普通节点完全一致，无需特判。
2. **快慢双指针制造间距**：`fast` 先走 `n` 步，使 `fast` 与 `slow` 之间恰好相隔 `n` 条边。然后 `fast` 与 `slow` 同速前进，当 `fast` 到达链表末尾（`fast.next == null`）时，`slow` 恰好停在待删节点的**前驱**。

![快慢双指针：fast 先走 n 步制造间距，再同速推进锁定前驱](images/remove_nth_fast_slow_pointers.svg)

**为什么慢指针正好停在前驱？**

- 初始：`fast` 先走 `n` 步，此时 `fast` 在第 `n+1` 个节点，`slow` 在 `dummy`（第 0 个），间距为 `n` 条边。
- 同速推进：每轮 `fast` 和 `slow` 各走 1 步，间距恒为 `n`。
- 终止：当 `fast.next == null`（`fast` 在最后一个节点）时，`slow` 与 `fast` 间距仍为 `n`。
- 设链表长度 `L`，则 `fast` 从位置 `n` 走到位置 `L`，走了 `L - n` 步。`slow` 从 `dummy`（位置 0）也走 `L - n` 步，到达位置 `L - n`——正好是倒数第 `n` 个节点的前驱（倒数第 `n` 个节点在位置 `L - n + 1`）。

> 💡 **核心洞察**：快指针的"先发优势"把"倒数第 n"这个后向参照系，转化为"领先 n 步"这个前向参照系。慢指针不用知道总长度，只需跟着快指针走——当快指针触底时，慢指针的位置就是答案。这是一次扫描的关键。

### 2.3 算法流程图

![算法流程：dummy → fast 先走 n 步 → 同速推进 → slow.next 跳过待删节点](images/remove_nth_algorithm_flow.svg)

**完整步骤**：

1. **建哑节点**：`dummy.next = head`，`slow = fast = dummy`
2. **快指针先发**：`fast` 向前走 `n` 步（此时 `fast` 与 `slow` 间距为 `n`）
3. **同速推进**：`while fast.next != null`：`slow = slow.next`，`fast = fast.next`
4. **删除**：`slow.next = slow.next.next`（跳过待删节点）
5. **返回**：`dummy.next`（真正的头节点）

> ⚠️ 循环条件是 `fast.next != null`（而非 `fast != null`），这样 `fast` 停在**最后一个节点**而非越过尾部，`slow` 恰好停在待删节点的前驱。若写成 `fast != null`，`slow` 会多走一步，停到待删节点本身，就无法删除了（删节点必须拿到前驱）。

### 2.4 示例演算

以 `head = [1,2,3,4,5]`，`n = 2` 为例：

![示例演算：fast 先走 2 步，再同速推进，slow 停在 3 删除 4](images/remove_nth_example_walkthrough.svg)

| 阶段 | slow 位置 | fast 位置 | 链表状态 | 说明 |
|------|----------|----------|----------|------|
| 初始 | dummy | dummy | 1→2→3→4→5 | 两指针都在哑节点 |
| fast 先走 2 步 | dummy | 2 | 1→2→3→4→5 | fast 走到节点 2，间距为 2 |
| 同速推进 1 | 1 | 3 | 1→2→3→4→5 | 各前进 1 步 |
| 同速推进 2 | 2 | 4 | 1→2→3→4→5 | 各前进 1 步 |
| 同速推进 3 | 3 | 5 | 1→2→3→4→5 | fast.next=null，停止 |
| 删除 | 3 | — | 1→2→3→5 | slow.next 指向 5，跳过 4 |

最终 `dummy.next = 1 → 2 → 3 → 5`。

> 💡 注意 `n = sz`（删除头节点）的情况：`head = [1,2,3,4,5]`, `n = 5`。`fast` 先走 5 步到达节点 5，同速推进时 `fast.next` 立即为 null，循环不执行，`slow` 仍在 `dummy`，`dummy.next = head.next` 即删除头节点 1。这正是哑节点的价值——无需特判头节点。

---

## 3. 参考代码

### C++

```cpp
// 删除链表的倒数第N个节点.cpp —— 哑节点 + 快慢双指针（一趟扫描）
// 编译: g++ -O2 -std=c++17 删除链表的倒数第N个节点.cpp -o remove_nth
struct ListNode {
    int val;
    ListNode* next;
    ListNode() : val(0), next(nullptr) {
    }
    ListNode(int x) : val(x), next(nullptr) {
    }
    ListNode(int x, ListNode* n) : val(x), next(n) {
    }
};

class Solution {
  public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode dummy(0); // 哑节点，统一处理删除头节点
        dummy.next = head;
        ListNode* slow = &dummy;
        ListNode* fast = &dummy;

        // 1. fast 先走 n 步，制造 n 条边的间距
        for (int i = 0; i < n; ++i) {
            fast = fast->next;
        }

        // 2. 同速推进，直到 fast 到达最后一个节点
        while (fast->next != nullptr) {
            slow = slow->next;
            fast = fast->next;
        }

        // 3. slow.next 是待删节点，跳过它
        ListNode* toDelete = slow->next;
        slow->next = slow->next->next;
        delete toDelete; // 释放内存（可选，C++ 良好习惯）

        return dummy.next;
    }
};
```

### Python

```python
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)           # 哑节点，统一处理删除头节点
        slow = fast = dummy

        # 1. fast 先走 n 步，制造 n 条边的间距
        for _ in range(n):
            fast = fast.next

        # 2. 同速推进，直到 fast 到达最后一个节点
        while fast.next:
            slow = slow.next
            fast = fast.next

        # 3. slow.next 是待删节点，跳过它
        slow.next = slow.next.next

        return dummy.next
```

> 💡 Python 中无需手动 `delete`，GC 自动回收。C++ 中 `delete toDelete` 是良好习惯，避免内存泄漏（但 LeetCode 的链表节点由系统管理，删不删都能 AC）。注意删除前保存 `toDelete` 指针，避免 `slow->next->next` 访问已释放内存。

---

## 4. 复杂度分析

| 维度 | 快慢双指针（一趟） | 两趟扫描 | 说明 |
|------|-------------------|---------|------|
| **时间复杂度** | `O(n)` | `O(n)` | 快慢指针各遍历一次，合计 `L` 步；两趟扫描合计 `2L` 步，同为 `O(n)` |
| **空间复杂度** | `O(1)` | `O(1)` | 只用 `dummy`、`slow`、`fast` 三个指针 |
| **扫描趟数** | **1 趟** | 2 趟 | 满足进阶"一趟扫描"要求 |

> ⚠️ 虽然两种方法时间复杂度都是 `O(n)`，但快慢双指针只扫描一趟，常数因子更优。更重要的是，一趟扫描体现了"双指针制造间距"这一通用思想，可迁移到"链表中点"、"找倒数第 k 个"等众多问题。

---

## 5. 扩展：双指针制造间距的通用模板

### 5.1 876 链表的中间节点

[876 题](https://leetcode.cn/problems/middle-of-the-linked-list/) 要求返回链表的中间节点。用**快慢双指针**——`slow` 走 1 步、`fast` 走 2 步，`fast` 到尾时 `slow` 即中点。这与本题的"制造间距"是双指针的两种典型用法：

| 题目 | 双指针模式 | 间距控制 |
|------|-----------|---------|
| 19 删除倒数第 N | 先发 + 同速 | `fast` 先走 `n` 步，锁定固定间距 |
| 876 链表中点 | 不同速 | `fast` 速度是 `slow` 的 2 倍，动态拉开间距 |
| 141 环形链表 | 不同速 | `fast` 速度是 `slow` 的 2 倍，追及判断 |

### 5.2 倒数第 k 个节点（不删除）

只需找到倒数第 `k` 个节点（不删除），逻辑与本题完全一致：哑节点 + `fast` 先走 `k` 步 + 同速推进。本题多了"删除"操作，因此用哑节点拿到前驱。

### 5.3 删除倒数第 N 个节点（扩展：删除倒数第 N 到第 M 个）

变体：一次删除倒数第 `n` 到第 `m` 个节点（`n <= m`）。思路：`fast` 先走 `m` 步锁定倒数第 `m` 的前驱 `slow1`，再用第二个慢指针从 `slow1` 回退 `m - n` 步锁定倒数第 `n` 的前驱 `slow2`，然后 `slow2.next = slow1.next.next` 整段跳过。

> 💡 双指针"制造间距"的核心是**用先发量锁定参照系**。无论"倒数第 k"、"中点"、"距尾 n 步"，本质都是让一个指针提前出发或走得更快，把"末尾参照"转化为"前向间距"。掌握这个模板，链表的定位类问题基本通吃。

---

## 6. 面试要点

1. **为什么要用哑节点（dummy node）？**

   - 不用哑节点时，删除头节点（`n == sz`）需要特判：直接 `return head.next`。这导致代码分叉，易漏边界。
   - 哑节点让"删除头节点"与"删除中间节点"逻辑统一——`slow` 始终有前驱（`dummy`），`slow.next = slow.next.next` 永远安全。
   - 这是链表题的**通用技巧**：凡涉及"删除/插入头节点可能变化"的场景，都用哑节点。最后返回 `dummy.next`。

2. **为什么循环条件是** `fast->next != null` **而不是** `fast != null`**？**

   - `fast != null` 会让 `fast` 越过尾部，`slow` 多走一步，停在**待删节点本身**而非前驱，无法删除（删节点必须拿到前驱）。
   - `fast->next != null` 让 `fast` 停在**最后一个节点**，`slow` 恰好停在待删节点的前驱。这是本题最易错的细节。
   - 记忆口诀："删节点找前驱，fast 停在倒数第一，slow 停在倒数第 n+1"。

3. **快指针先走 n 步后，为什么间距恰好是 n？**

   - `fast` 走 `n` 步，`slow` 未动，两者初始都在 `dummy`。所以 `fast` 相对 `slow` 领先 `n` 条边。
   - 同速推进时，每轮各走 1 步，相对距离恒定不变，始终为 `n`。
   - 当 `fast` 在最后一个节点时，`slow` 与 `fast` 间距 `n`，即 `slow` 在倒数第 `n+1` 个位置（前驱）。这是把"后向参照"转化为"前向间距"的关键。

4. `n == sz`**（删除头节点）时哑节点如何起作用？**

   - `fast` 先走 `n = sz` 步，到达最后一个节点。此时 `fast.next == null`，循环不执行，`slow` 仍在 `dummy`。
   - `slow.next = slow.next.next` 即 `dummy.next = head.next`，头节点被删除。
   - 无需任何 `if` 特判，逻辑与删除中间节点完全一致。这就是哑节点统一边界处理的价值。

5. **一趟扫描和两趟扫描，时间复杂度都是 O(n)，为什么还要一趟？**

   - 常数因子：两趟走 `2L` 步，一趟走 `L` 步（`fast` 的 `n` 步 + 同速的 `L - n` 步 = `L`）。
   - 更重要的是**思想迁移**：一趟扫描体现"双指针制造间距"，这是通用模板，可迁移到中点、倒数第 k、距尾 n 步等问题。面试官看重的是模板思维，而非省下的那 `L` 步。
   - 面试策略：先说两趟扫描的朴素思路，再优化为一趟扫描，展示从"知道长度才能定位"到"用间距锁定位置"的思维跃迁。

> 💡 **一句话总结**：删除倒数第 N 个节点是"快慢双指针制造间距"的招牌题——哑节点统一删除头节点的边界，`fast` 先走 `n` 步锁定间距，同速推进后 `slow` 恰停在前驱。一趟扫描 `O(n)` 时间、`O(1)` 空间。核心模板是"用先发量把后向参照转成前向间距"，可迁移到链表中点、倒数第 k、环形链表等所有"定位类"链表问题。

---

## 7. 同类练习题
- [19. 删除链表的倒数第 N 个节点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)：快慢双指针
- [83. 删除排序链表中的重复元素](https://leetcode.cn/problems/remove-duplicates-from-sorted-list/)：一次遍历
- [876. 链表的中间结点](https://leetcode.cn/problems/middle-of-the-linked-list/)：快慢指针
