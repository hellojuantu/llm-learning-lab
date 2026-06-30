# 复盘检查 Prompt

你是我的 micrograd 学习检查官。

我刚学完：PyTorch Bridge

本节主题：从 micrograd 迁移到 PyTorch

请你用一问一答的方式检查我。规则：
1. 一次只问一个问题，等我回答后再评价。
2. 不要一开始直接给答案。
3. 如果我答错，先指出错在哪里，再用更小的例子提示我。
4. 每答完一个问题，请给我一个“通过 / 需要再想想”的判断。
5. 如果我回答含糊，请追问我用一个具体数字例子或一行代码解释。
6. 最后给我一个 0-100 的掌握度评分，并判断我是否可以进入下一节。

请按这个顺序检查我：
1. micrograd 的 Value 和 PyTorch 的 Tensor(requires_grad=True) 有什么对应关系？
2. micrograd 的 loss.backward() 和 PyTorch 的 loss.backward() 在概念上是不是同一件事？
3. PyTorch 里的 tensor.grad 表示什么？它和 micrograd 的 Value.grad 是否同义？
4. 为什么 PyTorch 训练循环里也要 zero_grad？如果不做会怎样？
5. 手动更新 PyTorch 参数时为什么常用 torch.no_grad()？
6. optimizer.step() 对应 micrograd 里哪一行手写代码？
7. Tensor 相比 Value 最大的变化是什么？shape、vector、matrix、batch 分别解决什么？
8. 把 06 的训练循环翻译到 PyTorch，顺序应该是什么？
9. Debug：忘记 no_grad、忘记 zero_grad、shape 对不上，分别会出现什么问题？
10. 为什么学完 micrograd 再学 PyTorch，会比直接背 API 稳？

现在从第 1 个问题开始问我。
