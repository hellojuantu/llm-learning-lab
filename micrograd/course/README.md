# micrograd Course

这里是课程模板。学习时不要直接改这里的 notebook，先复制到本地工作区：

```bash
cd micrograd
python3 sync_personal.py
```

然后打开：

```text
personal/course/START_HERE.ipynb
```

模板入口是 [START_HERE.ipynb](./START_HERE.ipynb)，用于查看和同步，不作为日常填写区。

第一阶段只盯住一件事：

```text
前向算 data -> loss 衡量错多少 -> backward 算 grad -> 按 -grad 更新参数 -> loss 下降
```

每节包含：

```text
preview.ipynb    课前预习作业，先填 TODO，再跑 qa_check 暴露卡点
class.ipynb      上课板书/实验，先跑代码，再拆概念
homework.ipynb   正式作业，按理解台阶组织编号任务、测试、Debug Lab、项目题
review_prompt.md 学完后用来复盘
```

## 练习方式

题不追求多，重点是每一步都能跑、能改、能检查。大多数作业会按这个顺序走：

```text
先跑一个小例子
-> 改一个条件
-> 自己补 TODO
-> 处理一个边界或反例
-> 修一个常见错误
-> 接回训练流程
```

做完一节后，至少要能回答：

```text
这题在练什么？
上一个例子的做法，哪里被复用了？
如果 qa_check 失败，错在数学、Python、计算图，还是训练循环？
这一节怎么接回 loss / grad / backward / update？
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

开始学习前同步到个人工作区：

```bash
python3 sync_personal.py
```

默认不会覆盖已经写过的 `personal/course` 文件。需要重置某一节时再显式使用：

```bash
python3 sync_personal.py --lesson 08_pytorch_bridge --force
```

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
python3 course/smoke_solutions.py
```

脚本通过不代表课程就没问题。每次改完材料，都要像学生一样亲自读对应的 `preview.ipynb`、`class.ipynb`、`homework.ipynb` 和 `review_prompt.md`：

```text
题目说明是否看得懂？
TODO 是否知道要填什么？
提示是否真的能救人？
答案是否只在展开后出现？
这一节能不能顺着读下来，而不是只靠检查器猜答案？
```

脚本只负责挡住格式错误、旧输出、检查器回退和明显坏味道。课程有没有教会人，必须靠人读。

旧的 v1 quick labs 已退休并删除；现在从 `course/` 同步到 `personal/course/` 后再学习。

更多设计原则见 [DESIGN_PHILOSOPHY.md](./DESIGN_PHILOSOPHY.md)。
