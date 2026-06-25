# micrograd Full Course Design

这份设计是 micrograd 课程的 v2 蓝图。它不是给现有 notebook 补几个小题，而是按小瓜/萧井陌式课程重新拆成完整学习工件。

## 设计依据

从 Xiao 课程材料里抽出来的硬标准：

```text
预习任务 -> 上课板书/实验 -> 正式作业文件 -> 测试 -> 作业讲解/调试 -> pro/结构课
```

正式作业不是零散小测，而是一组围绕理解障碍设计的递进任务：

```text
完整例子
-> 照例子改
-> 换一个条件
-> 处理边界
-> 抽成参数
-> 复用上一题
-> 接入真实流程
-> 故意出错并调试
```

每节课都必须让学习者在一个可运行工件里工作，而不是只读解释。

## 设计理念：不是题量，是理解台阶

小瓜式作业最重要的不是“多”，而是每一步都刚好拆掉一个真实卡点。

因此本课程不使用“每节必须多少题”当核心标准。题数可以变，但每节必须有一条清楚的理解台阶：

```text
不会看见问题
-> 能在小例子里看见问题
-> 能模仿解决
-> 能改一个条件
-> 能处理边界
-> 能解释错误
-> 能把这个能力接回项目主线
```

对 micrograd 来说，真正的主线只有一条：

```text
data 前向算值
-> loss 衡量错多少
-> backward 算 dL/d参数
-> 参数沿 -grad 更新
-> loss 下降
```

所有题都要服务这条主线。如果某道题不能帮助学生理解 `data`、`grad`、`backward`、`loss`、`update`、`shape`、`debug` 中的某一个关键连接，就应该删掉或改掉。

## 课程目录形态

v2 课程不要继续把所有东西塞进一个 notebook。每一课拆成一组文件：

```text
course/
  00_math_bootcamp/
    preview.ipynb
    class.ipynb
    homework.ipynb
    review_prompt.md
  01_gradient_homework/
    preview.ipynb
    class.ipynb
    homework.ipynb
    review_prompt.md
  ...
  11_final_project/
    class.ipynb
    homework.ipynb
    defense.md
```

文件职责：

```text
preview.ipynb    课前预习，10 分钟暴露卡点
class.ipynb      上课讲义/实验，先跑、再拆、再解释
homework.ipynb   正式作业，按理解台阶组织编号任务，带测试
review_prompt.md AI 复盘检查，不直接给答案
defense.md       项目答辩清单
```

## 作业文件标准

每个 `homework.ipynb` 必须包含：

```text
1. 本次作业用到的最小资料
2. 一个完整可运行例子
3. ensure / qa_check 测试函数
4. 足够覆盖关键理解台阶的编号作业
5. 每题都有函数/变量骨架
6. 每题都有测试
7. 后一题明确复用哪一题
8. 覆盖本节关键误区的 Debug Lab
9. 能把本节知识接回主线的小项目接入题
10. AI 作业讲解 prompt
```

难度梯度示例：

```text
作业 1：照例子改数字
作业 2：参考作业 1，换一个符号
作业 3：参考作业 2，处理负数/0/重复变量
作业 4：把固定公式抽成函数
作业 5：把函数接到 micrograd Value
作业 6：写测试用例
作业 7：修一个故意写错的版本
作业 8：解释为什么测试失败
作业 9：接入完整训练/项目流程
```

## Full Course Map

### 00 - Math Bootcamp

角色：`class + homework`

目标：把导数、偏导、梯度、链式法则变成可手算的小工具。

预习：

```text
一组很小的题：斜率、固定变量、方向变化率、简单链式法则。
不会写的题先记下来，上课重点听。
```

上课：

```text
从 y = 3x+1 的斜率开始
画二维函数里 x/y 两个方向
把 L = 2(ab+a) 拆成路径图
用表格算路径贡献
```

正式作业：

```text
作业 1：算 y=2x+1 的斜率
作业 2：参考作业 1，算 y=-3x+4
作业 3：固定 b，算 L=ab+a 对 a 的影响
作业 4：固定 a，算 L=ab+a 对 b 的影响
作业 5：把 L=2(ab+a) 写成 u=ab+a, L=2u
作业 6：算 y=(3x+1)^2
作业 7：路径表：a -> c -> L
作业 8：多路径：a -> c -> d -> L 和 a -> d -> L
Debug 1：把 ReLU 值和 ReLU 导数混了
Debug 2：多路径只算了一条
项目题：解释 micrograd 里的 grad 为什么是 dL/dValue
```

### 01 - Gradient Homework

角色：`homework`

目标：大量手算，让链式法则形成肌肉记忆。

正式作业：

```text
作业 1：L=a+b
作业 2：参考作业 1，L=a-b
作业 3：L=a*b
作业 4：参考作业 3，L=a*a
作业 5：L=a*b+a
作业 6：参考作业 5，L=2(a*b+a)
作业 7：L=(a+b)^2
作业 8：L=(2w+1)^2
作业 9：L=relu(w)
作业 10：参考作业 9，w<0 时 relu
作业 11：多路径相加
作业 12：负数路径
作业 13：画路径表
Debug 1：把乘法局部梯度写反
Debug 2：把 += 写成 =
项目题：手算 02 里 Value 例子的 grad
```

### 02 - Value Usage

角色：`class`

目标：不看源码，只从使用者视角理解 `data`、`grad`、`backward`。

上课：

```text
a = Value(2.0)
b = Value(3.0)
c = a*b
d = c+a
L = d*2
```

正式作业：

```text
作业 1：写出 a,b,c,d,L 的 data
作业 2：参考作业 1，改 a=4
作业 3：backward 前 grad 为什么是 0
作业 4：backward 后 a.grad/b.grad
作业 5：解释 a.grad=8
作业 6：解释 b.grad=4
作业 7：手算和 micrograd 对照
作业 8：重复 backward 观察累加
作业 9：重新建图后再 backward
Debug 1：把 grad 当成参数变化量
Debug 2：同一张图重复 backward
项目题：写一个打印 data/grad 的小函数
```

### 03 - Value Source Reading

角色：`class + source reading`

目标：读懂 `Value.__init__`、`__add__`、`__mul__`、闭包、`backward`。

正式作业：

```text
作业 1：c=a+b 时 self/other/out 分别是谁
作业 2：参考作业 1，c=a*3 时 other 是谁
作业 3：普通数字为什么包装成 Value
作业 4：out._prev 和 out._op
作业 5：手动调用加法 _backward
作业 6：乘法 _backward
作业 7：闭包记住了哪些变量
作业 8：L=a+a 为什么必须 +=
作业 9：最终节点 grad 为什么是 1
作业 10：topo 顺序合法性判断
作业 11：set 去重为什么重要
Debug 1：for child in v._prev 的语法错误
Debug 2：topo.append 写成 topo.add
项目题：自己写一个只有加法的 TinyValue
```

### 04 - TinyValue Homework

角色：`homework`

目标：完整实现一个小型自动求导类。

正式作业：

```text
作业 1：__init__
作业 2：__repr__
作业 3：__add__ 前向
作业 4：__add__ 反向
作业 5：__mul__ 前向
作业 6：__mul__ 反向
作业 7：__pow__
作业 8：relu
作业 9：__neg__/__sub__
作业 10：__radd__/__rmul__
作业 11：build_topo
作业 12：backward
作业 13：通过 L=2(ab+a)
作业 14：通过重复变量 L=a*a
Debug 1：pow 漏乘 out.grad
Debug 2：relu 值和导数混淆
Debug 3：缩进错误
项目题：通过 grade_value_class
```

### 05 - Neuron And MLP

角色：`class + shape homework`

目标：把标量自动求导接到神经元、Layer、MLP。

正式作业：

```text
作业 1：区分数学 [] 和 Python []
作业 2：点积 w*x
作业 3：sum(zip(...), b) 展开
作业 4：Neuron(2) 参数数量
作业 5：参考作业 4，Neuron(3)
作业 6：Layer(2,3) 展开成 3 个 Neuron(2)
作业 7：Layer(3,1)
作业 8：MLP(2,[3,1]) 展开
作业 9：参数数量 13
作业 10：backward 后参数 grad 的含义
Debug 1：bias 当成每层一个
Debug 2：把 ReLU(x) 写成 x>0 时为 1
项目题：画出 MLP(2,[3,1]) 的计算图
```

### 06 - Training Loop Project

角色：`pro`

目标：闭合最小训练循环。

正式作业：

```text
作业 1：手算一条样本的平方误差
作业 2：参考作业 1，扩展到 4 条样本
作业 3：写 predict
作业 4：写 loss
作业 5：zero_grad
作业 6：backward
作业 7：更新一个参数
作业 8：更新所有参数
作业 9：跑 1 step
作业 10：跑 50 step
作业 11：画 loss 曲线
作业 12：调 learning_rate
Debug 1：忘记 zero_grad
Debug 2：更新方向写成 +grad
Debug 3：learning_rate 太大
项目题：训练一个小回归模型并解释 loss 为什么下降
```

### 07 - Classification Boundary Project

角色：`pro + visualization`

目标：从回归训练过渡到二分类、margin、hinge loss 和 decision boundary。

正式作业：

```text
作业 1：score -> label
作业 2：参考作业 1，score 接近 0
作业 3：margin = y*score
作业 4：hinge loss
作业 5：答对但不够远
作业 6：答错
作业 7：画红点蓝点
作业 8：训练前 score 表
作业 9：训练后 score 表
作业 10：画 score=0 黑线
作业 11：解释点离黑线远近
作业 12：改 margin 从 1 到 2
Debug 1：把 hinge loss 当 accuracy
Debug 2：把 ReLU 值写成 1
项目题：训练并解释一个二维分类边界图
```

### 08 - PyTorch Bridge

角色：`structure/translation`

目标：把 micrograd 词汇迁移到 PyTorch。

正式作业：

```text
作业 1：Value -> Tensor 对照
作业 2：requires_grad
作业 3：loss.backward
作业 4：tensor.grad
作业 5：zero_grad
作业 6：no_grad
作业 7：手动更新 w
作业 8：optimizer.step
作业 9：shape 观察
作业 10：batch 观察
作业 11：把 06 的训练循环翻译成 PyTorch
作业 12：对照 micrograd 和 PyTorch 的 loss 曲线
Debug 1：忘记 no_grad
Debug 2：忘记 zero_grad
项目题：写一个最小 PyTorch 训练循环
```

### 09 - Gradient Debugging

角色：`debugging homework`

目标：训练失败时能定位问题。

正式作业：

```text
作业 1：重复 backward
作业 2：忘记 zero_grad
作业 3：grad 全是 0
作业 4：grad 爆大
作业 5：learning_rate 太大
作业 6：learning_rate 太小
作业 7：ReLU dead
作业 8：loss 不下降但 accuracy 变
作业 9：参数没更新
作业 10：更新方向反了
作业 11：数据标签错了
作业 12：画 debug checklist
Debug 1-6：每题一个故意坏掉的训练循环
项目题：给一个坏训练脚本，写出诊断报告
```

### 10 - Mini Project

角色：`pro + checklist`

目标：完成一个可解释的小分类项目。

正式作业：

```text
作业 1：定义数据
作业 2：画数据
作业 3：建模型
作业 4：写 predict
作业 5：写 loss
作业 6：写 accuracy
作业 7：写训练循环
作业 8：画 loss 曲线
作业 9：画 decision boundary
作业 10：列出错分样本
作业 11：解释一个参数 grad
作业 12：写项目复盘
Debug 1：分界线图和 accuracy 冲突
Debug 2：loss 下降但图没变
项目题：最终答辩
```

### 11 - Structure And Next Route

角色：`structure + roadmap`

目标：把 micrograd 项目结构、PyTorch、D2L、LLM 路线接起来。

正式作业：

```text
作业 1：画 micrograd 文件结构
作业 2：engine.py 负责什么
作业 3：nn.py 负责什么
作业 4：demo 负责什么
作业 5：PyTorch 对应什么
作业 6：D2L 要补什么
作业 7：Happy-LLM 入口是什么
作业 8：LLMs-from-scratch 主链路是什么
Debug 1：把会用库误当成理解原理
Debug 2：资源清单过载
项目题：写下一阶段学习计划
```

## 验收标准

课程改造完成后，不用“看起来更详细”作为标准，而用下面检查：

```text
每个 homework 的编号作业必须覆盖本节关键理解台阶
Debug Lab 必须覆盖本节最容易犯的真实错误
项目接入题必须把本节知识接回 micrograd 主线
每个作业有测试
后一个作业明确复用前一个作业
空白状态 Run All 不崩
填错时错误信息指出概念问题
每节有 AI 作业讲解 prompt
pro/structure 课要在学习者开始觉得知识碎片化之前出现
```

## 与旧 v1 quick labs 的关系

旧的 v1 quick labs 已退休并从仓库中删除，避免和正式课程入口混淆。

v2 课程已经落到 `course/00-11`：

```text
preview.ipynb   课前小练习
class.ipynb     课堂实验，先预测、再运行、再改一行
homework.ipynb  正式作业，包含 TODO、qa_check、Debug Lab、项目接入
review_prompt.md 课后 AI 复盘
```

后续优化不再往 `course/` 里堆 Markdown 说明，而是优先改 notebook 里的可运行练习、测试和调试任务。
