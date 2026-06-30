"""Run reference solutions for every learner-facing course check.

This is stricter than executing blank notebooks.  It fills each exercise with a
known-good answer and requires the corresponding ``qa_check`` to actually pass.
"""

from __future__ import annotations

import contextlib
import io
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable


COURSE = Path(__file__).resolve().parent
ROOT = COURSE.parent
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from course.checks import qa_check  # noqa: E402
from micrograd.engine import Value  # noqa: E402
from micrograd.nn import Layer, MLP  # noqa: E402

try:
    import torch
except Exception as exc:  # pragma: no cover - this course requires torch here
    raise RuntimeError("PyTorch must be installed for solution smoke tests.") from exc


RAN_CHECKS: set[str] = set()


def notebook_check_names() -> set[str]:
    names: set[str] = set()
    for path in sorted([*COURSE.glob("*.ipynb"), *COURSE.glob("*/*.ipynb")]):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            marker = "qa_check('"
            start = 0
            while True:
                index = source.find(marker, start)
                if index == -1:
                    break
                index += len(marker)
                end = source.find("'", index)
                names.add(source[index:end])
                start = end + 1
    return names


def run(name: str, ns: dict[str, Any] | None = None, *args: Any) -> str:
    ns = ns or {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        qa_check(name, ns, *args)
    text = output.getvalue()
    if "OK:" not in text:
        raise AssertionError(f"{name} did not pass with OK output. Output:\n{text}")
    bad_fragments = ["请先", "请继续", "请返回", "还不能", "还没过", "跳过"]
    for fragment in bad_fragments:
        if fragment in text:
            raise AssertionError(f"{name} printed learner-facing failure text:\n{text}")
    RAN_CHECKS.add(name)
    return text


def micrograd_one_step() -> tuple[float, float]:
    w = Value(2.0)
    x = Value(3.0)
    y = Value(7.0)
    loss = (w * x - y) ** 2
    loss.backward()
    grad = w.grad
    w.data -= 0.1 * w.grad
    return grad, w.data


def torch_one_step() -> tuple[float, float]:
    w = torch.tensor(2.0, requires_grad=True)
    x = torch.tensor(3.0)
    y = torch.tensor(7.0)
    loss = (w * x - y) ** 2
    loss.backward()
    grad = float(w.grad.item())
    with torch.no_grad():
        w -= 0.1 * w.grad
    return grad, float(w.item())


def torch_one_step_full() -> tuple[float, float, float, float]:
    w = torch.tensor(2.0, requires_grad=True)
    x = torch.tensor(3.0)
    y = torch.tensor(7.0)
    pred = w * x
    loss = (pred - y) ** 2
    loss.backward()
    grad = float(w.grad.item())
    with torch.no_grad():
        w -= 0.1 * w.grad
    return float(pred.item()), float(loss.item()), grad, float(w.item())


class MyTinyValue:
    def __init__(self, data: float, children: tuple[Any, ...] = (), op: str = ""):
        self.data = data
        self.grad = 0.0
        self._prev = set(children)
        self._op = op
        self._backward: Callable[[], None] = lambda: None

    def __add__(self, other: Any):
        other = other if isinstance(other, MyTinyValue) else MyTinyValue(other)
        out = MyTinyValue(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Any):
        other = other if isinstance(other, MyTinyValue) else MyTinyValue(other)
        out = MyTinyValue(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: float):
        out = MyTinyValue(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = MyTinyValue(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other


def source_add_object_probe() -> tuple[float, str, bool, bool]:
    a = Value(2.0)
    b = Value(3.0)
    out = a + b
    return out.data, out._op, a in out._prev, b in out._prev


def source_mul_wrap_probe() -> tuple[float, str, int]:
    a = Value(2.0)
    out = a * 3
    return out.data, out._op, len(out._prev)


def source_add_probe() -> tuple[tuple[float, str, int], tuple[float, str, int]]:
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    d = a + 3
    return (c.data, c._op, len(c._prev)), (d.data, d._op, len(d._prev))


def source_backward_probe() -> tuple[float, float]:
    a = Value(2.0)
    L = a + a
    L.backward()
    return a.grad, L.grad


def build_topo_demo(root: Value) -> list[Value]:
    topo: list[Value] = []
    visited: set[Value] = set()

    def build(v: Value) -> None:
        if v in visited:
            return
        visited.add(v)
        for child in v._prev:
            build(child)
        topo.append(v)

    build(root)
    return topo


def repeat_backward_demo() -> tuple[float, float, float]:
    a = Value(2.0)
    b = Value(3.0)
    L = (a * b + a) * 2
    L.backward()
    first = a.grad
    L.backward()
    second = a.grad
    a2 = Value(2.0)
    b2 = Value(3.0)
    L2 = (a2 * b2 + a2) * 2
    L2.backward()
    return first, second, a2.grad


def layer_bias_probe() -> tuple[int, int, int]:
    layer = Layer(2, 3)
    return len(layer.neurons), sum(1 for n in layer.neurons for p in [n.b]), len(layer.parameters())


def training_ns() -> dict[str, Any]:
    ns: dict[str, Any] = {
        "xs": [[0.0], [1.0], [2.0], [3.0]],
        "ys": [1.0, 3.0, 5.0, 7.0],
        "w": Value(0.0),
        "b": Value(0.0),
    }

    def predict(xrow):
        return ns["w"] * xrow[0] + ns["b"]

    def loss():
        preds = [predict(x) for x in ns["xs"]]
        losses = [(pred - y) ** 2 for pred, y in zip(preds, ns["ys"])]
        return sum(losses) * (1.0 / len(losses))

    def train_step(learning_rate=0.1):
        ns["w"].grad = 0
        ns["b"].grad = 0
        L = loss()
        before = L.data
        L.backward()
        ns["w"].data -= learning_rate * ns["w"].grad
        ns["b"].data -= learning_rate * ns["b"].grad
        return before

    ns.update({"predict": predict, "loss": loss, "train_step": train_step})
    return ns


def label_from_score(score: float) -> int:
    return 1 if score > 0 else -1


def margin(score: Value, y: int) -> Value:
    return y * score


def hinge_loss(score: Value, y: int) -> Value:
    return (1 - y * score).relu()


def torch_grad_accumulation_demo() -> tuple[float, float, float]:
    w = torch.tensor(2.0, requires_grad=True)
    x = torch.tensor(3.0)
    y = torch.tensor(7.0)
    ((w * x - y) ** 2).backward()
    first = float(w.grad.item())
    ((w * x - y) ** 2).backward()
    second = float(w.grad.item())
    w.grad.zero_()
    ((w * x - y) ** 2).backward()
    after_zero = float(w.grad.item())
    return first, second, after_zero


def torch_update_and_batch_demo() -> tuple[float, tuple[int, ...], list[float]]:
    updated_w = torch_one_step()[1]
    batch = torch.tensor([1.0, 2.0, 3.0])
    doubled = batch * 2
    return updated_w, tuple(batch.shape), [float(x) for x in doubled.tolist()]


def zero_grad_demo(scale: float) -> tuple[float, float, float]:
    x = Value(2.0)
    (x * scale).backward()
    first = x.grad
    (x * scale).backward()
    second = x.grad
    x.grad = 0
    (x * scale).backward()
    return first, second, x.grad


def update_direction_pair() -> tuple[float, float]:
    old_w, grad, lr = 5.0, 8.0, 0.1
    return old_w + lr * grad, old_w - lr * grad


def debug_symptom_probe() -> tuple[float, float, float, float, float]:
    return 6, 5.8, 4.2, 0, 0


def fixed_update(p: Value, lr: float) -> None:
    p.data -= lr * p.grad


def mini_project_ns() -> dict[str, Any]:
    random.seed(7)
    ns: dict[str, Any] = {
        "xs": [(-2, -1), (-1, -2), (-2, 1), (-1, 1), (1, 2), (2, 1), (1, -1), (2, -2), (0.5, 1.5), (-1.5, -0.5)],
        "ys": [-1, -1, -1, -1, 1, 1, 1, 1, 1, -1],
        "model": MLP(2, [4, 1]),
    }

    def predict_score(xrow):
        return ns["model"]([Value(x) for x in xrow])

    def predict_label(xrow):
        return 1 if predict_score(xrow).data > 0 else -1

    def project_hinge_loss(score, y):
        return (1 - y * score).relu()

    def total_loss():
        losses = [project_hinge_loss(predict_score(x), y) for x, y in zip(ns["xs"], ns["ys"])]
        return sum(losses) * (1.0 / len(losses))

    def accuracy():
        return sum(predict_label(x) == y for x, y in zip(ns["xs"], ns["ys"])) / len(ns["ys"])

    def train(steps=30, learning_rate=0.05):
        history = []
        for _ in range(steps):
            ns["model"].zero_grad()
            L = total_loss()
            history.append(L.data)
            L.backward()
            for p in ns["model"].parameters():
                p.data -= learning_rate * p.grad
        return history

    ns.update(
        {
            "predict_score": predict_score,
            "predict_label": predict_label,
            "hinge_loss": project_hinge_loss,
            "total_loss": total_loss,
            "accuracy": accuracy,
            "train": train,
        }
    )
    return ns


def structure_probe() -> tuple[bool, bool, bool]:
    engine = ROOT / "micrograd" / "engine.py"
    nn = ROOT / "micrograd" / "nn.py"
    demo = ROOT / "demo.ipynb"
    return "class Value" in engine.read_text(), "class MLP" in nn.read_text(), demo.exists()


def update_bridge_demo() -> tuple[float, float]:
    manual = torch_one_step()[1]
    w = torch.tensor(2.0, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.1)
    loss = (w * torch.tensor(3.0) - torch.tensor(7.0)) ** 2
    loss.backward()
    opt.step()
    return manual, float(w.item())


def next_route() -> list[dict[str, str]]:
    return [
        {"stage": "PyTorch", "pass_question": "能用 Tensor 写出 micrograd 同款 forward/backward/update 吗？"},
        {"stage": "D2L", "pass_question": "能解释线性回归、MLP、反向传播和优化器各解决什么问题吗？"},
        {"stage": "Happy-LLM", "pass_question": "能把 Transformer 流程从 token 讲到 logits 吗？"},
        {"stage": "LLMs-from-scratch", "pass_question": "能手写一个最小 GPT 并跑通 generate 吗？"},
        {"stage": "nanoGPT", "pass_question": "能读懂真实训练循环、checkpoint 和 batch 数据加载吗？"},
    ]


def run_all() -> None:
    run("qa_check_start_here", {"student_value_observation": lambda: (6, 0, 3)})
    run("qa_check_preview_00", {}, 2)
    run("qa_check_class_00_predict", {"predict_slope": 2})
    run("qa_check_00_modify", {"f_modify": lambda x: -3 * x + 4, "student_modify_slope": -3})
    run("qa_check_class_00_modify", {"student_dy_dx": 42})
    run("qa_check_math_bootcamp", {"student_slope_1": 2, "student_slope_2": -3, "student_dL_da": 4, "student_dL_db": 2, "student_chain": 42})
    run("qa_check_math_debug", {"student_relu_value_at_3": 3, "student_repeated_path_grad": 2})

    run("qa_check_preview_01", {}, 3, 2)
    run("qa_check_class_01_predict", {"predict_dL_da": 3, "predict_dL_db": 2})
    run("qa_check_01_modify", {"student_modify_dL_da": 4, "student_modify_dL_db": 2})
    run("qa_check_class_01_modify", {"student_dL_da": 4, "student_dL_db": 2})
    run("qa_check_gradient_practice", {"student_grad_L_eq_a_plus_b": 1, "student_grad_L_eq_a_times_b_da": 3, "student_grad_L_eq_a_times_b_db": 2, "student_grad_L_eq_a_square": 4, "student_grad_L_eq_2ab_plus_2a_da": 8, "student_grad_L_eq_2ab_plus_2a_db": 4, "student_grad_L_eq_2w_plus_1_square": 28})
    run("qa_check_repeated_variable_explain", {"student_repeated_variable_paths": lambda a: (a, a, 2 * a)})

    run("qa_check_preview_02", {}, 6, 8, 16)
    run("qa_check_class_02_predict", {"predict_values": [2, 3, 6, 8, 16]})
    run("qa_check_02_modify", {"Value": Value, "student_modify_L_data": 32, "student_modify_a_grad": 8, "student_modify_b_grad": 8})
    run("qa_check_class_02_modify", {"Value": Value, "student_L_data": 32, "student_a_grad": 8, "student_b_grad": 8})
    value_ns = {"Value": Value}
    value_ns["a"], value_ns["b"] = Value(2.0), Value(3.0)
    value_ns["c"] = value_ns["a"] * value_ns["b"]
    value_ns["d"] = value_ns["c"] + value_ns["a"]
    value_ns["L"] = value_ns["d"] * 2
    value_ns["student_data"] = [2, 3, 6, 8, 16]
    value_ns["student_grads"] = [8, 4, 2, 2, 1]
    run("qa_check_value_usage", value_ns)
    run("qa_check_repeat_backward", {"student_repeat_backward_demo": repeat_backward_demo})

    run("qa_check_preview_03", {"student_add_object_probe": source_add_object_probe})
    run("qa_check_class_03_predict", {"student_add_object_probe": source_add_object_probe})
    run("qa_check_class_03_modify", {"student_mul_wrap_probe": source_mul_wrap_probe})
    run("qa_check_03_modify", {"student_mul_probe": source_mul_wrap_probe})
    run("qa_check_source_reading_basic", {"student_add_probe": source_add_probe})
    run("qa_check_source_reading_backward", {"student_backward_probe": source_backward_probe})
    run("qa_check_source_debug", {"Value": Value, "student_build_topo_demo": build_topo_demo})

    run("qa_check_preview_04", {}, 2.0, 0.0)
    run("qa_check_class_04_predict", {"student_minivalue_state": lambda: ((2.0, 0.0), (set(), "", lambda: None))})
    run("qa_check_class_04_modify", {"student_add_backward_rule": lambda out_grad: (out_grad, out_grad)})
    run("qa_check_04_modify", {"student_add_backward_rule": lambda out_grad: (out_grad, out_grad)})
    run("qa_check_04_forward", {}, MyTinyValue)
    run("qa_check_04_backward", {}, MyTinyValue)
    run("qa_check_04_debug", {"student_pow_relu_debug_values": lambda: (30, 3, 1)})

    run("qa_check_preview_05", {}, 3, 1, 4)
    run("qa_check_class_05_predict", {"student_neuron3": [3, 1, 4]})
    run("qa_check_05_modify", {"student_neuron3_weight_count": 3, "student_neuron3_bias_count": 1, "student_neuron3_output_count": 1})
    run("qa_check_class_05_modify", {"student_param_count": 13})
    run("qa_check_neuron_mlp", {"student_dot": -3, "student_neuron3_shape": [3, 1, 4], "student_layer23": [3, 2], "student_mlp_layers": [(2, 3), (3, 1)], "student_mlp_param_count": 13})
    run("qa_check_bias_debug", {"student_layer_bias_probe": layer_bias_probe})

    run("qa_check_preview_06", {}, 4.2)
    run("qa_check_class_06_predict", {"student_new_p": 4.2})
    run("qa_check_class_06_modify", {"student_new_p_small_lr": 4.92})
    run("qa_check_06_modify", {"student_new_p_with_small_lr": 4.92})
    run("qa_check_training_intro", {"student_training_intro_probe": lambda: (1.0, 4.2)})
    run("qa_check_train_step", training_ns())
    run("qa_check_update_direction", {"student_correct_new_w": 4.2})

    run("qa_check_preview_07", {}, 0.2, 2.0)
    run("qa_check_class_07_predict", {"student_margin": 2})
    run("qa_check_class_07_modify", {"student_loss_margin2": 0.5})
    run("qa_check_07_hinge", {"Value": Value, "label_from_score": label_from_score, "margin": margin, "hinge_loss": hinge_loss})
    run("qa_check_07_modify", {"student_hinge_margin2": 0.5})
    run("qa_check_07_debug", {"student_classification_debug_cases": lambda: (0.8, 3, 2)})

    run("qa_check_preview_08", {}, torch.tensor(2.0, requires_grad=True))
    run("qa_check_class_08_predict", {"student_torch_one_step": torch_one_step_full})
    run("qa_check_class_08_modify", {"student_grad_accumulation_demo": torch_grad_accumulation_demo})
    run("qa_check_08_mapping", {"student_compare_micrograd_torch": lambda: (micrograd_one_step(), torch_one_step())})
    run("qa_check_08_modify", {"student_w_after_lr_001": 2.06})
    run(
        "qa_check_08_hand_then_run",
        {
            "student_manual_pred": 6.0,
            "student_manual_loss": 1.0,
            "student_manual_grad": -6.0,
            "student_manual_updated_w": 2.6,
            "student_torch_hand_check": torch_one_step_full,
        },
    )
    run("qa_check_08_debug", {"student_torch_update_and_batch_demo": torch_update_and_batch_demo})

    run("qa_check_preview_09", {"student_zero_grad_demo": lambda: zero_grad_demo(2)})
    run("qa_check_class_09_predict", {"student_grad_accumulation_probe": lambda: zero_grad_demo(3)})
    run("qa_check_class_09_modify", {"student_fixed_update_demo": update_direction_pair})
    run("qa_check_09_modify", {"student_compare_update_directions": update_direction_pair})
    run("qa_check_debug_mapping", {"student_debug_symptom_probe": debug_symptom_probe})
    run("qa_check_fixed_update_line", {}, fixed_update)

    run("qa_check_preview_10", {}, ["forward", "loss", "backward", "update"])
    run("qa_check_class_10_predict", {"student_order": ["forward", "loss", "backward", "update"]})
    run("qa_check_class_10_modify", {"student_param_count": 13})
    run("qa_check_10_modify", {"student_param_count_mlp_2_3_1": 13})
    project = mini_project_ns()
    run("qa_check_10_project_functions", project)
    run("qa_check_10_train", project)
    run("qa_check_10_debug", {"student_project_debug_probe": lambda: (True, 0, 0)})

    run("qa_check_preview_11", {}, lambda: (True, True))
    run("qa_check_class_11_predict", {"student_structure_probe": lambda: (True, True, True)})
    run("qa_check_class_11_modify", {"student_update_bridge_demo": update_bridge_demo})
    run("qa_check_11_file_roles", {"student_inspect_micrograd_files": structure_probe})
    run("qa_check_11_torch_mapping", {"student_torch_bridge_probe": lambda: (micrograd_one_step(), torch_one_step())})
    run("qa_check_11_modify", {"student_update_bridge_demo": update_bridge_demo})
    run("qa_check_11_next_route", {}, next_route())
    run("qa_check_11_debug", {"student_route_debug": lambda: (False, True, False)})

    expected = notebook_check_names()
    missing = expected - RAN_CHECKS
    extra = RAN_CHECKS - expected
    if missing:
        raise AssertionError(f"smoke_solutions.py did not run notebook checks: {sorted(missing)}")
    if extra:
        raise AssertionError(f"smoke_solutions.py ran checks not present in notebooks: {sorted(extra)}")
    print(f"OK solution smoke: ran {len(RAN_CHECKS)} learner checks.")


if __name__ == "__main__":
    run_all()
