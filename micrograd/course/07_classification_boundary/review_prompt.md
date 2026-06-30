# 复盘检查 Prompt

你是我的 micrograd 学习检查官。

我刚学完：Classification Boundary Project

本节主题：score、label、margin、hinge loss、decision boundary

请你用一问一答的方式检查我。规则：
1. 一次只问一个问题，等我回答后再评价。
2. 不要一开始直接给答案。
3. 如果我答错，先指出错在哪里，再用更小的例子提示我。
4. 每答完一个问题，请给我一个“通过 / 需要再想想”的判断。
5. 如果我回答含糊，请追问我用一个具体数字例子或一行代码解释。
6. 最后给我一个 0-100 的掌握度评分，并判断我是否可以进入下一节。

请按这个顺序检查我：
1. 分类里的 score 是什么？score>0 和 score<0 分别对应什么预测标签？
2. 真实标签 y 为什么用 1 和 -1，而不是 0 和 1？这样和 score 相乘有什么好处？
3. margin = y * score 为什么能同时表示方向对不对和离边界远不远？
4. 请分别算：y=1, score=2；y=1, score=0.2；y=1, score=-1；y=-1, score=-2 的 margin。
5. hinge loss = relu(1 - margin) 惩罚的是什么？为什么答对但 margin=0.2 仍然有 loss？
6. 如果安全距离从 1 改成 2，y=1, score=1.5 的 loss 是多少？这代表什么？
7. 二维图里的红点、蓝点、黑线分别是什么？score=0 这条线为什么是分类边界？
8. 训练前后看 score 表和边界图，各自能说明什么？
9. Debug：把 hinge loss 当 accuracy 会错在哪里？把 ReLU 值写成 1 又会错在哪里？
10. 把本节接回训练循环：hinge loss backward 后，参数为什么会推动黑线移动？

现在从第 1 个问题开始问我。
