# micrograd Full Course v2

这是按小瓜/萧井陌课程形态重新设计的 micrograd full course。每一节都是一个文件夹，而不是一个孤立 notebook。

第一入口：[START_HERE.ipynb](./START_HERE.ipynb)

这个入口假设学习者不懂 LLM、不懂 PyTorch、不懂完整深度学习，只先跑通一条主链：

```text
前向算 data -> loss 衡量错多少 -> backward 算 grad -> 按 -grad 更新参数 -> loss 下降
```

每节固定包含：

```text
preview.ipynb    课前预习作业，先填 TODO，再跑 qa_check 暴露卡点
class.ipynb      上课板书/实验，先跑代码，再拆概念
homework.ipynb   正式作业，按理解台阶组织编号任务、测试、Debug Lab、项目题
review_prompt.md AI 复盘检查 prompt
```

课程设计标准：

```text
材料/资料 -> 完整例子 -> 编号作业 -> 骨架/TODO -> 实现步骤 -> 测试
-> 调试/作业讲解 -> 下一题复用上一题 -> pro/结构课
```

## 设计理念

题的数量不是目标，学生真的跨过理解障碍才是目标。

这里的作业不是为了凑“很多题”，而是按下面的顺序搭桥：

```text
先暴露一个具体卡点
-> 用小数字例子跑通
-> 改一个条件
-> 处理一个边界
-> 抽象成规则
-> 接回 micrograd 主线
-> 故意坏掉一次并修好
```

所以每节课验收的不是“做了几题”，而是：

```text
学生能不能说出这一题在练什么
学生能不能把上一题的做法迁移到下一题
学生能不能通过测试知道自己错在哪里
学生能不能把局部知识接回 loss / grad / backward / update 主链路
```

## Lessons

- [00_math_bootcamp](./00_math_bootcamp/) - Math Bootcamp (class + homework)
- [01_gradient_homework](./01_gradient_homework/) - Gradient Homework (homework)
- [02_value_usage](./02_value_usage/) - Value Usage (class)
- [03_value_source_reading](./03_value_source_reading/) - Value Source Reading (class + source reading)
- [04_tinyvalue_homework](./04_tinyvalue_homework/) - TinyValue Homework (homework)
- [05_neuron_mlp](./05_neuron_mlp/) - Neuron And MLP (class + shape homework)
- [06_training_loop](./06_training_loop/) - Training Loop Project (pro)
- [07_classification_boundary](./07_classification_boundary/) - Classification Boundary Project (pro + visualization)
- [08_pytorch_bridge](./08_pytorch_bridge/) - PyTorch Bridge (structure/translation)
- [09_gradient_debugging](./09_gradient_debugging/) - Gradient Debugging (debugging homework)
- [10_mini_project](./10_mini_project/) - Mini Project (pro + checklist)
- [11_structure_next_route](./11_structure_next_route/) - Structure And Next Route (structure + roadmap)

## Local Environment

不要把 PyTorch / Jupyter 依赖装到系统 Python。用本地 venv：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r course/requirements-course.txt
.venv/bin/python -m ipykernel install --user --name micrograd-course --display-name "micrograd-course"
```

然后在 Jupyter 里选择 `micrograd-course` kernel。

## Pass Criteria

```text
每个 homework 的编号作业必须覆盖本节关键理解台阶
后一个作业必须明确复用前一个作业或前面规则
Debug Lab 必须覆盖本节最容易犯的真实错误
项目接入题必须把本节知识接回 micrograd 主线
每个作业最终都要有 qa_check / assert / 可执行检查
空白状态 Run All 不崩，填错时错误信息指出概念问题
```

课程质量审计：

```bash
python3 course/audit_course.py
```

旧的 v1 quick labs 已退休并删除；这个 `course/` 是当前 full course 设计入口。

更多设计原则见 [DESIGN_PHILOSOPHY.md](./DESIGN_PHILOSOPHY.md)。
