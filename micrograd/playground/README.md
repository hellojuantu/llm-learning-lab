# micrograd 无痛学习路线

这套 notebook 的目标不是“看完源码”，而是把一条主线真正练熟：

```text
前向计算 -> loss -> 导数/梯度 -> backward -> 参数 grad -> 参数更新 -> loss 下降
```

课程按“萧井陌 + 吴恩达”风格改造：

```text
课前预习作业 -> 上课讲义/实验 -> 课后作业检查 -> AI 复盘
```

也就是先用小题动手，再用讲义补原理，最后用测试确认自己真的会。

## 怎么启动

从 `llm` 根目录启动 Jupyter：

```bash
cd /Users/barry/IdeaProjects/llm
./start_notebook.sh
```

然后打开：

```text
micrograd/playground/
```

## 学习规则

每节都按这个节奏走：

```text
1. 先做“课前预习作业”，哪怕只会猜
2. 再读“一句话心智模型”
3. 先预测输出，不要急着运行
4. 运行代码
5. 改一行，再运行
6. 做 TODO 或 qa_check
7. 对照“课后作业提交清单”
8. 最后发送 AI 复盘 prompt 检查理解
```

不要一路 `Run All` 当看电影。你要停下来猜、算、改、错一次。

## 萧井陌式三段结构

参考 [ProjectOpenCourse](https://github.com/artisanbox/ProjectOpenCourse) 的组织方式：课前先做小作业，上课再补变量、循环、函数这些概念；参考 [xiaogua-python-homework](https://github.com/artisanbox/xiaogua-python-homework) 的作业形态：同一个作业文件里给资料、函数骨架、测试和编号任务；参考 [XiaoGua](https://github.com/artisanbox/XiaoGua) 的长期课程形态：`class`、`homework`、`pro`、`结构`、`知识清单` 分阶段推进。

所以这套 micrograd 课程每本都改成三段：

```text
课前预习作业：先用小数字做 1 个最小任务
上课讲义/实验：看图、公式、源码或运行结果
课后作业检查：用提交清单确认能不能进入下一节
```

这不是让你先学完所有理论再写代码，而是先做一个小东西，卡住的地方再回到讲义补基础。

映射到当前 micrograd 第一阶段：

```text
00-03：class/讲义课，负责补关键概念和源码理解
01、04：homework/作业课，负责大量手算和完整 TinyValue 实现
05-06：pro/小项目课，把 Value 接到 MLP，再跑通训练闭环
每节末尾：知识清单，用来判断能不能进入下一节
```

## 练习题格式

所有练习题尽量按同一种格式：

```text
题目 -> 留空作答 cell -> qa_check 测试 -> Show / Hide 提示 -> Show / Hide 答案
```

做题时先把 `None` 或 `pass` 改成你的答案，再运行 `qa_check...`。

```python
your_a_grad = None
qa_check('ex1', your_a_grad, your_b_grad)
```

如果还没填写，测试会温和提示；如果填错，会明确告诉你哪里不对。提示和答案是分开的：先看提示，最后再看答案。

## 教学课和练习课的边界

这套课程不是每一本都做完整实现。边界是：

```text
00：数学直觉课，带少量小测
01：手算练习课
02：Value 使用观察课，带少量小测
03：Value 源码理解课，只做局部补全和出口检查
04：TinyValue 完整实现作业课
05：Neuron / MLP 结构理解课，带少量小测
06：训练循环理解课，带少量小测
```

所以 `03` 和 `04` 不重复：

```text
03 负责回答：micrograd 的 Value 源码为什么这么写？
04 负责训练：我能不能自己写出一个 TinyValue？
```

如果你在 `03` 里已经想写完整实现，可以先忍一下，去 `04` 写；这样学习节奏更稳。

## 推荐顺序

### 00 - `00_math_review.ipynb`

补 micrograd 需要的最小数学。

你要带走：

```text
导数：一元函数某一点的斜率
偏导：多元函数里，只让一个变量变时的斜率
梯度：所有偏导组成的向量
链式法则：局部变化率沿路径相乘
```

过关标准：

```text
看到 L = 2(ab+a)，能手算 dL/da=2(b+1), dL/db=2a。
```

### 01 - `01_gradient_practice.ipynb`

专门练手算梯度。

你要带走：

```text
加法：梯度原样传
乘法：梯度交叉乘
平方：先设中间变量
多路径：路径贡献要相加
ReLU：正数传，负数断
```

过关标准：

```text
不看答案，能算对前 5 题；再用自动检查确认 10 题都通过。
```

### 02 - `02_scalar_graph.ipynb`

只从使用者视角观察 `Value`。

这一节不读源码，只回答：

```text
Value 怎么前向算 L.data？
L.backward() 后 grad 是什么？
grad 和手算偏导是否一致？
```

过关标准：

```text
能解释 a.grad=8、b.grad=4 分别就是 dL/da、dL/db。
```

### 03 - `03_value_implementation.ipynb`

开始读 `micrograd.engine.Value` 的实现。

你要带走：

```text
Value(data) 是带梯度和来源信息的数字
+ 和 * 会创建新 Value 节点
每个新节点保存父节点、运算符和局部 _backward
backward() 先拓扑排序，再倒序调用 _backward
```

过关标准：

```text
能把 c = a + b 代入 self=a, other=b, out=c 讲清楚。
能解释为什么 +=、为什么 L.grad 先设成 1、为什么需要 topo。
```

### 04 - `04_tinyvalue_exercises.ipynb`

把 `Value` 的核心自己写出来。

你要完成：

```text
补全加法反传
补全乘法反传
补全 __pow__
补全 relu
补全 backward 拓扑排序
通过 grade_value_class
```

过关标准：

```text
能自己写一个支持 +、*、**、relu、backward 的 TinyValue。
```

### 05 - `05_neuron_and_mlp.ipynb`

把 `Value` 接到神经网络结构。

这一节开头先补最小够用的线性代数：

```text
数学里的 [] = 向量/矩阵的排版符号
Python 里的 [] = list 容器，里面装一组对象
向量 = 一排数，比如输入 x 和权重 w
点积 = 对应位置相乘再相加
bias = 最后额外加上的一个可学习标量
Layer = 很多个神经元，也就是很多个点积并排算
```

你要带走：

```text
Neuron = 点积 w*x + b + ReLU
Layer = 多个 Neuron
MLP = 多个 Layer
参数 = w 和 b，它们也是 Value
```

过关标准：

```text
能区分数学里的 [x1, x2] 和 Python 里的 x = [x1, x2]。
能解释 sum((wi*xi for wi,xi in zip(w, x)), b) 为什么是 w*x+b。
能算出 MLP(2, [3, 1]) 为什么有 13 个参数。
```

### 06 - `06_training_loop.ipynb`

闭合训练主线。

你要带走：

```text
forward 算预测
loss 衡量错误
zero_grad 清空旧梯度
backward 算新梯度
p.data += -learning_rate * p.grad 更新参数
重复多轮让 loss 下降
```

过关标准：

```text
能解释为什么更新方向是 -grad，以及 learning_rate 控制每步走多大。
```

## 每节后的 AI 复盘

每本 notebook 末尾都有：

```text
AI 复盘检查 Prompt
```

学完一节后，把它发送给任意 AI，让它一问一答检查你。

建议你这样回复 AI：

```text
先不要给我完整答案。
如果我答错，只提示我下一步该看哪个变量。
等我连续两次卡住，再给局部答案。
```

## 学习记录模板

每学完一节，在自己的笔记里写 5 行：

```text
今天学的是：
一句话总结：
我最容易混淆的是：
我跑通的代码/测试：
下一节开始前我要复习：
```

## 当前阶段不要纠结

先不要纠结：

```text
matplotlib 每一行画图 API
Jupyter 高级调试技巧
PyTorch 内部实现
大型数据集训练
复杂优化器 Adam
```

当前只抓一条线：

```text
为什么 loss.backward() 能算出每个参数的 grad？
```

把这条线走通，后面再学 PyTorch、Transformer、LLM，都会稳很多。
