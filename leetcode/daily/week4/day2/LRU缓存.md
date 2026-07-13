# LRU 缓存

- **题目名称**：LRU 缓存
- **链接**：[146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)
- **难度**：中等
- **标签**：设计、哈希表、链表、双向链表

## 1. 题目概述

设计并实现满足 **LRU（Least Recently Used，最近最少使用）** 缓存约束的数据结构。需要支持以下操作且均摊时间复杂度为 `O(1)`：

- `LRUCache(int capacity)`：以**正整数** `capacity` 作为缓存容量初始化。
- `int get(int key)`：如果关键字 `key` 存在于缓存中，则返回关键字的值，**并标记为最近使用**；否则返回 `-1`。
- `void put(int key, int value)`：如果关键字 `key` 已经存在，则变更其数据值 `value`，**并标记为最近使用**；如果不存在，则向缓存中插入该组 `key-value`。如果插入操作导致关键字数量超过 `capacity`，则应该**逐出最久未使用**的关键字。

> 💡 **LRU 的核心语义**：访问（读或写）过的元素被认为是「最近使用」的，会被提升到最前；最久未被访问的元素位于最后，容量满时优先被淘汰。

**示例 1**：

```text
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1，缓存是 {2=2, 1=1}
lRUCache.put(3, 3); // 逐出键 2，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1（未找到）
lRUCache.put(4, 4); // 逐出键 1，缓存是 {3=3, 4=4}
lRUCache.get(1);    // 返回 -1（未找到）
lRUCache.get(3);    // 返回 3，缓存是 {4=4, 3=3}
lRUCache.get(4);    // 返回 4，缓存是 {3=3, 4=4}
```

**约束条件**：

- `1 <= capacity <= 3000`
- `0 <= key <= 10^4`
- `0 <= value <= 10^5`
- 最多调用 `2 * 10^5` 次 `get` 和 `put`

> ⚠️ **进阶**：你是否可以实现 `O(1)` 时间复杂度完成 `get` 和 `put` 操作？

---

## 2. 解题思路

### 2.1 暴力思路

最朴素的方案是用一个**数组或单向链表**按访问时间从新到旧排列：

- `get(key)`：线性扫描找到元素，返回值，再把该元素**移动到表头**。时间复杂度 `O(n)`。
- `put(key, value)`：线性查找是否存在；若存在则更新并移到表头；若不存在则插入表头，超容量时删除表尾。时间复杂度 `O(n)`。

**瓶颈**：每次「查找」和「移动到表头」都是 `O(n)`，对于 `2 * 10^5` 次调用会达到 `O(10^10)`，必然超时。

### 2.2 核心观察：哈希表 + 双向链表

要同时满足「`O(1)` 查找」和「`O(1)` 调整顺序」，需要把两套数据结构组合起来：

- **哈希表** `unordered_map<int, Node*>`：`key → 链表节点指针`，负责 `O(1)` 定位。
- **双向链表**：节点带 `prev` / `next` 指针，负责 `O(1)` 删除与插入（无需找前驱）。
- **哨兵节点**：用 `dummy_head`（最近使用）和 `dummy_tail`（最久未使用）两个虚拟节点夹住真实节点，**消除边界判空**。

![LRU 缓存数据结构](images/lru_cache_structure.svg)

约定：**`dummy_head` 端是最近使用（MRU）**，**`dummy_tail` 端是最久未使用（LRU）**。

- `get(key)`：哈希表查到节点 → 把节点**移到 `dummy_head` 之后** → 返回值。
- `put(key, value)`：
  - 命中：更新 `value`，移到表头。
  - 未命中：新建节点插入表头，登记到哈希表；若超容量，则**删除 `dummy_tail` 前一个节点**并从哈希表移除。

> 💡 **为什么必须双向链表？** 删除尾节点（淘汰）时需要拿到它的前驱；单向链表只能 `O(n)` 找前驱，双向链表的 `prev` 指针让删除一步到位，从而保证 `O(1)`。

### 2.3 算法流程图

![LRU get/put 流程](images/lru_cache_flow.svg)

两个操作的核心都是「**定位 + 摘出 + 接到表头**」三步，其中「摘出」和「接到表头」是双向链表的基础操作，固定为指针改写。

### 2.4 示例演算

![LRU 示例演算](images/lru_cache_walkthrough.svg)

以示例 1 中 `capacity = 2` 为例：

| 步骤 | 操作 | 链表（MRU→LRU） | 说明 |
|------|------|------------------|------|
| 1 | put(1,1) | 1 | 插入表头 |
| 2 | put(2,2) | 2→1 | 插入表头，容量未满 |
| 3 | get(1) | 1→2 | 命中，移到表头，返回 1 |
| 4 | put(3,3) | 3→1 | 容量满，淘汰 2，再插入 3 |
| 5 | get(2) | 3→1 | 未命中，返回 -1 |
| 6 | put(4,4) | 4→3 | 容量满，淘汰 1，再插入 4 |

---

## 3. 参考代码

### C++（哈希表 + 双向链表 + 哨兵节点）

```cpp
struct Node {
    int key, val;
    Node* prev;
    Node* next;
    Node(int k, int v) : key(k), val(v), prev(nullptr), next(nullptr) {
    }
};

class LRUCache {
  private:
    int cap;
    unordered_map<int, Node*> mp;
    Node* head; // 哑节点，dummy_head（MRU 端）
    Node* tail; // 哑节点，dummy_tail（LRU 端）

    // 把节点摘出链表
    void remove(Node* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    // 把节点插入到 dummy_head 之后（成为最近使用）
    void addToHead(Node* node) {
        node->next = head->next;
        node->prev = head;
        head->next->prev = node;
        head->next = node;
    }

    // 把已存在的节点提升为最近使用
    void moveToHead(Node* node) {
        remove(node);
        addToHead(node);
    }

    // 淘汰最久未使用（dummy_tail 前一个）
    Node* evictTail() {
        Node* node = tail->prev;
        remove(node);
        return node;
    }

  public:
    LRUCache(int capacity) : cap(capacity) {
        head = new Node(0, 0);
        tail = new Node(0, 0);
        head->next = tail;
        tail->prev = head;
    }

    int get(int key) {
        auto it = mp.find(key);
        if (it == mp.end())
            return -1;
        Node* node = it->second;
        moveToHead(node);
        return node->val;
    }

    void put(int key, int value) {
        auto it = mp.find(key);
        if (it != mp.end()) {
            Node* node = it->second;
            node->val = value;
            moveToHead(node);
            return;
        }
        Node* node = new Node(key, value);
        mp[key] = node;
        addToHead(node);
        if ((int)mp.size() > cap) {
            Node* removed = evictTail();
            mp.erase(removed->key);
            delete removed;
        }
    }
};
```

### Python（手写双向链表版，面试推荐）

```python
class Node:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.mp = {}  # key -> Node
        self.head = Node()  # 哑节点（MRU 端）
        self.tail = Node()  # 哑节点（LRU 端）
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_head(self, node: Node) -> None:
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def _move_to_head(self, node: Node) -> None:
        self._remove(node)
        self._add_to_head(node)

    def get(self, key: int) -> int:
        if key not in self.mp:
            return -1
        node = self.mp[key]
        self._move_to_head(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.mp:
            node = self.mp[key]
            node.val = value
            self._move_to_head(node)
            return
        node = Node(key, value)
        self.mp[key] = node
        self._add_to_head(node)
        if len(self.mp) > self.cap:
            removed = self.tail.prev
            self._remove(removed)
            del self.mp[removed.key]
```

### Python（`OrderedDict` 简洁版，工程快速实现）

```python
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.od = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.od:
            return -1
        self.od.move_to_end(key)  # 移到末尾表示最近使用
        return self.od[key]

    def put(self, key: int, value: int) -> None:
        if key in self.od:
            self.od.move_to_end(key)
        self.od[key] = value
        if len(self.od) > self.cap:
            self.od.popitem(last=False)  # 弹出头部 = 最久未使用
```

> ⚠️ **面试策略**：`OrderedDict` 版本能秒过 OJ，但**面试官通常希望你手写双向链表**以验证对指针操作的掌握。建议先讲清楚手写版思路，再提及 `OrderedDict` 作为工程上的简洁替代。

---

## 4. 复杂度分析

| 维度 | 复杂度 | 说明 |
|------|--------|------|
| `get` 时间 | O(1) | 哈希表查找 + 双向链表指针改写（常数次） |
| `put` 时间 | O(1) | 哈希表插入 + 链表插入 / 淘汰（常数次指针操作） |
| 空间复杂度 | O(capacity) | 哈希表与链表最多存 `capacity` 个节点 |

---

## 5. 扩展：LFU 缓存与 LRU 变体

- **LFU 缓存（[460. LFU 缓存](https://leetcode.cn/problems/lfu-cache/)）**：在 LRU 基础上改为「**访问频率最低**者优先淘汰，频率相同时按 LRU」。需要额外维护 `freq → 节点链表` 的映射，以及全局 `min_freq`，实现难度更高。
- **LRU 变体**：
  - **带 TTL 的 LRU**：节点存过期时间戳，`get` 时惰性删除；或后台定期清理。
  - **LRU-K**：记录每个 key 最近 K 次访问，只有访问满 K 次才进入主缓存，用于数据库缓冲池。
  - **分段 LRU（如 Redis 的近似 LRU）**：避免维护全局链表的高开销，随机采样若干候选 key 淘汰最久未用者。

> 💡 Linux 内核的 page cache、Redis 的 `maxmemory-policy allkeys-lru`、MySQL buffer pool 都基于 LRU 的变体，理解 146 题是读懂这些系统的入口。

---

## 6. 面试要点

1. **为什么需要双向链表而不是单向链表？**

   - 淘汰最久未使用节点时，需要拿到它的**前驱**来改指针。单向链表找前驱是 `O(n)`，双向链表通过 `prev` 指针一步 `O(1)` 完成。

2. **为什么要用 `dummy_head` / `dummy_tail` 两个哨兵节点？**

   - 消除「链表为空」「插入/删除的是头尾节点」等边界判空逻辑，让 `addToHead`、`remove` 等函数对任意节点都走同一套指针改写，代码更短、更不容易出 bug。

3. **`put` 时容量超限，为什么先插入再淘汰也能保证正确？**

   - 先插入会让 `size = capacity + 1`，此时淘汰 `tail.prev` 后回到 `capacity`。由于新插入的节点在表头，淘汰的必然是之前最久未使用的节点，语义和「先淘汰再插入」等价。

4. **C++ 中为什么要 `delete` 被淘汰的节点？**

   - `new` 出来的节点不释放会内存泄漏。在 `2 * 10^5` 次调用下泄漏可能很明显。生产代码中也可用 `unique_ptr` 或智能指针托管，但面试中手写 `delete` 更直观。

5. **如果要求线程安全怎么做？**

   - 给整个缓存加一把读写锁（`shared_mutex`）是最简单的方案，但粒度粗、并发度低；工业级实现（如 `concurrent-lru`）通常采用**分片（sharding）**：把 key 哈希到多个独立的 LRU 桶，每个桶一把锁，降低争用。

---

## 7. 同类练习题
- [460. LFU 缓存](https://leetcode.cn/problems/lfu-cache/)：LFU 设计
- [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)：LRU 设计
- [380. O(1) 时间插入、删除和获取随机元素](https://leetcode.cn/problems/insert-delete-getrandom-o1/)：设计题
