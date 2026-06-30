"""Course exercise feedback helpers.

The notebooks call ``qa_check(name, globals(), *args)``.  This module keeps the
learner-facing cells small, but the checks are normal Python functions rather
than hidden answer strings.  The goal is Xiao-style feedback: run the learner's
artifact, point at the next missing step, and accept behavior instead of exact
phrasing whenever possible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


CHECKS: dict[str, Callable[..., Any]] = {}


def check(name: str):
    def decorator(fn: Callable[..., Any]):
        CHECKS[name] = fn
        return fn

    return decorator


def near(actual: Any, expected: Any, eps: float = 1e-6) -> bool:
    return abs(float(actual) - float(expected)) < eps


def _contains_todo(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, Mapping):
        return any(_contains_todo(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_todo(v) for v in value)
    return False


def todo_guard(values: Iterable[Any], message: str = "请先完成 TODO，再运行检查。") -> bool:
    if any(_contains_todo(v) for v in values):
        print(message)
        return False
    return True


def _get(ns: dict[str, Any], name: str) -> Any:
    return ns.get(name)


def _student_fn(ns: dict[str, Any], *names: str) -> Callable[..., Any] | None:
    for name in names:
        value = ns.get(name)
        if callable(value):
            return value
    print("请先完成 TODO：还没有找到要检查的 student 函数。")
    return None


def _run(fn: Callable[..., Any], *args: Any) -> Any:
    try:
        result = fn(*args)
    except Exception as exc:
        print("代码还不能跑：", type(exc).__name__, exc)
        return None
    if _contains_todo(result):
        print("请先完成 TODO：函数还返回了 None。")
        return None
    return result


def _as_dict(result: Any) -> dict[str, Any] | None:
    if isinstance(result, Mapping):
        return dict(result)
    print("请返回一个 dict，这样每个中间结果都能被检查。")
    return None


def _as_tuple(result: Any, length: int) -> tuple[Any, ...] | None:
    if isinstance(result, (list, tuple)) and len(result) == length:
        return tuple(result)
    print(f"请返回长度为 {length} 的 tuple/list。")
    return None


def _assert_close(actual: Any, expected: Any, name: str, eps: float = 1e-6) -> None:
    assert near(actual, expected, eps), f"{name}: expected {expected}, got {actual}"


def _require_keys(data: Mapping[str, Any], keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in data]
    assert not missing, f"返回结果缺少 key: {missing}"


def _value_class(ns: dict[str, Any]):
    value_cls = ns.get("Value")
    if value_cls is not None:
        return value_cls
    from micrograd.engine import Value

    return Value


def _module_root(ns: dict[str, Any]) -> Path:
    root = ns.get("ROOT")
    if root is not None:
        return Path(root)
    return Path(__file__).resolve().parents[1]


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


@check("qa_check_start_here")
def qa_check_start_here(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_value_observation")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    _require_keys(data, ["data_before", "grad_before", "grad_after"])
    _assert_close(data["data_before"], 6.0, "y.data")
    _assert_close(data["grad_before"], 0.0, "x.grad before backward")
    _assert_close(data["grad_after"], 3.0, "x.grad after backward")
    print("OK: 你已经用运行结果区分了 data 和 grad。")


@check("qa_check_preview_00")
def qa_check_preview_00(ns: dict[str, Any], value: Any) -> None:
    if not todo_guard([value]):
        return
    _assert_close(value, 2, "slope")
    print("OK: 一元导数=局部斜率。")


@check("qa_check_class_00_predict")
def qa_check_class_00_predict(ns: dict[str, Any]) -> None:
    qa_check_preview_00(ns, ns.get("predict_slope"))


@check("qa_check_00_modify")
def qa_check_00_modify(ns: dict[str, Any]) -> None:
    f_modify = ns.get("f_modify")
    slope = ns.get("student_modify_slope")
    if not callable(f_modify) or not todo_guard([slope]):
        return
    if f_modify(0) is None:
        print("请先完成 TODO：f_modify 还没有返回直线值。")
        return
    measured = (f_modify(2.0001) - f_modify(2)) / 0.0001
    _assert_close(measured, -3, "measured slope", 1e-3)
    _assert_close(slope, -3, "student_modify_slope")
    print("OK: Modify 通过。")


@check("qa_check_class_00_modify")
def qa_check_class_00_modify(ns: dict[str, Any]) -> None:
    value = ns.get("student_dy_dx")
    if not todo_guard([value]):
        return
    _assert_close(value, 42, "dy/dx")
    print("OK: 链式法则 Modify 通过。")


@check("qa_check_math_bootcamp")
def qa_check_math_bootcamp(ns: dict[str, Any]) -> None:
    values = [
        ns.get("student_slope_1"),
        ns.get("student_slope_2"),
        ns.get("student_dL_da"),
        ns.get("student_dL_db"),
        ns.get("student_chain"),
    ]
    if not todo_guard(values):
        return
    expected = [2, -3, 4, 2, 42]
    for actual, exp, name in zip(values, expected, ["slope1", "slope2", "dL/da", "dL/db", "chain"]):
        _assert_close(actual, exp, name)
    print("OK: 数学基础台阶通过。")


@check("qa_check_math_debug")
def qa_check_math_debug(ns: dict[str, Any]) -> None:
    relu_value = ns.get("student_relu_value_at_3")
    repeated_grad = ns.get("student_repeated_path_grad")
    if not todo_guard([relu_value, repeated_grad]):
        return
    _assert_close(relu_value, 3, "ReLU(3)")
    _assert_close(repeated_grad, 2, "d(a+a)/da")
    print("OK: 数学 Debug 通过。")


@check("qa_check_preview_01")
def qa_check_preview_01(ns: dict[str, Any], da: Any, db: Any) -> None:
    if not todo_guard([da, db]):
        return
    _assert_close(da, 3, "d(a*b)/da")
    _assert_close(db, 2, "d(a*b)/db")
    print("OK: 乘法局部梯度通过。")


@check("qa_check_class_01_predict")
def qa_check_class_01_predict(ns: dict[str, Any]) -> None:
    qa_check_preview_01(ns, ns.get("predict_dL_da"), ns.get("predict_dL_db"))


@check("qa_check_01_modify")
def qa_check_01_modify(ns: dict[str, Any]) -> None:
    da = ns.get("student_modify_dL_da")
    db = ns.get("student_modify_dL_db")
    if not todo_guard([da, db]):
        return
    _assert_close(da, 4, "d(ab+a)/da")
    _assert_close(db, 2, "d(ab+a)/db")
    print("OK: Modify 通过。")


@check("qa_check_class_01_modify")
def qa_check_class_01_modify(ns: dict[str, Any]) -> None:
    da = ns.get("student_dL_da")
    db = ns.get("student_dL_db")
    if not todo_guard([da, db]):
        return
    _assert_close(da, 4, "dL/da")
    _assert_close(db, 2, "dL/db")
    print("OK: 多路径 Modify 通过。")


@check("qa_check_gradient_practice")
def qa_check_gradient_practice(ns: dict[str, Any]) -> None:
    names = [
        "student_grad_L_eq_a_plus_b",
        "student_grad_L_eq_a_times_b_da",
        "student_grad_L_eq_a_times_b_db",
        "student_grad_L_eq_a_square",
        "student_grad_L_eq_2ab_plus_2a_da",
        "student_grad_L_eq_2ab_plus_2a_db",
        "student_grad_L_eq_2w_plus_1_square",
    ]
    values = [ns.get(name) for name in names]
    if not todo_guard(values):
        return
    for name, actual, expected in zip(names, values, [1, 3, 2, 4, 8, 4, 28]):
        _assert_close(actual, expected, name)
    print("OK: 梯度手算通过。")


@check("qa_check_repeated_variable_explain")
def qa_check_repeated_variable_explain(ns: dict[str, Any], fn: Callable[[float], Any] | None = None) -> None:
    fn = fn or _student_fn(ns, "student_repeated_variable_paths")
    if fn is None:
        return
    for a_value in [2.0, -3.0, 0.5]:
        result = _as_tuple(_run(fn, a_value), 3)
        if result is None:
            return
        left, right, total = result
        _assert_close(left, a_value, "left path")
        _assert_close(right, a_value, "right path")
        _assert_close(total, 2 * a_value, "total grad")
    print("OK: 你用两条路径算出了 L=a*a 的梯度。")


@check("qa_check_preview_02")
def qa_check_preview_02(ns: dict[str, Any], c_data: Any, d_data: Any, L_data: Any) -> None:
    if not todo_guard([c_data, d_data, L_data]):
        return
    for actual, expected, name in [(c_data, 6, "c.data"), (d_data, 8, "d.data"), (L_data, 16, "L.data")]:
        _assert_close(actual, expected, name)
    print("OK: 前向 data 预测通过。")


@check("qa_check_class_02_predict")
def qa_check_class_02_predict(ns: dict[str, Any]) -> None:
    values = ns.get("predict_values")
    if not todo_guard([values]):
        return
    assert list(values) == [2, 3, 6, 8, 16], f"实际应为 [2,3,6,8,16]，got {values}"
    print("OK: data 预测通过。")


@check("qa_check_02_modify")
def qa_check_02_modify(ns: dict[str, Any]) -> None:
    values = [ns.get("student_modify_L_data"), ns.get("student_modify_a_grad"), ns.get("student_modify_b_grad")]
    if not todo_guard(values):
        return
    Value = _value_class(ns)
    a = Value(4.0)
    b = Value(3.0)
    L = (a * b + a) * 2
    L.backward()
    for actual, expected, name in zip(values, [L.data, a.grad, b.grad], ["L.data", "a.grad", "b.grad"]):
        _assert_close(actual, expected, name)
    print("OK: Modify 通过。")


@check("qa_check_class_02_modify")
def qa_check_class_02_modify(ns: dict[str, Any]) -> None:
    values = [ns.get("student_L_data"), ns.get("student_a_grad"), ns.get("student_b_grad")]
    if not todo_guard(values):
        return
    Value = _value_class(ns)
    a = Value(4.0)
    b = Value(3.0)
    L = (a * b + a) * 2
    L.backward()
    for actual, expected, name in zip(values, [L.data, a.grad, b.grad], ["L.data", "a.grad", "b.grad"]):
        _assert_close(actual, expected, name)
    print("OK: 改输入后仍能解释 data/grad。")


@check("qa_check_value_usage")
def qa_check_value_usage(ns: dict[str, Any]) -> None:
    student_data = ns.get("student_data")
    student_grads = ns.get("student_grads")
    if not todo_guard([student_data, student_grads]):
        return
    L = ns.get("L")
    if L is None:
        print("请先运行上面的前向计算。")
        return
    L.backward()
    actual_data = [ns["a"].data, ns["b"].data, ns["c"].data, ns["d"].data, ns["L"].data]
    actual_grads = [ns["a"].grad, ns["b"].grad, ns["c"].grad, ns["d"].grad, ns["L"].grad]
    assert list(student_data) == actual_data, f"data 实际是 {actual_data}"
    assert list(student_grads) == actual_grads, f"grad 实际是 {actual_grads}"
    print("OK: Value 使用层 data/grad 通过。")


@check("qa_check_repeat_backward")
def qa_check_repeat_backward(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_repeat_backward_demo")
    if fn is None:
        return
    result = _as_tuple(_run(fn), 3)
    if result is None:
        return
    first, second, fresh = result
    _assert_close(first, 8, "first backward grad")
    assert second > first, f"第二次 backward 后 grad 应该继续累加变大，got first={first}, second={second}"
    _assert_close(fresh, 8, "fresh graph grad")
    print("OK: 你跑出了重复 backward 的累加问题。")


def _check_add_probe_dict(data: dict[str, Any]) -> None:
    _require_keys(data, ["out_data", "out_op", "a_is_parent", "b_is_parent"])
    _assert_close(data["out_data"], 5, "out.data")
    assert data["out_op"] == "+", f"out._op 应该是 '+', got {data['out_op']!r}"
    assert data["a_is_parent"] is True and data["b_is_parent"] is True, "a/b 都应该在 out._prev 里。"


@check("qa_check_preview_03")
def qa_check_preview_03(ns: dict[str, Any], fn: Callable[[], Any] | None = None, *legacy: Any) -> None:
    if legacy:
        if not todo_guard([fn, *legacy]):
            return
        assert [fn, *legacy] == ["a", "b", "c"]
        print("OK: Python 魔法方法代入通过。")
        return
    fn = fn or _student_fn(ns, "student_add_object_probe")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    _check_add_probe_dict(data)
    print("OK: 你用对象状态看清了 a+b 的 out/_prev/_op。")


@check("qa_check_class_03_predict")
def qa_check_class_03_predict(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_add_object_probe")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    _check_add_probe_dict(data)
    print("OK: self/other/out 已经用运行结果代清楚了。")


@check("qa_check_class_03_modify")
def qa_check_class_03_modify(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_mul_wrap_probe")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    _require_keys(data, ["out_data", "out_op", "parent_count"])
    _assert_close(data["out_data"], 6, "out.data")
    assert data["out_op"] == "*", f"out._op 应该是 '*', got {data['out_op']!r}"
    assert data["parent_count"] == 2, "a*3 包装后也应该有两个父节点。"
    print("OK: 普通数字包装成 Value 的逻辑通过。")


@check("qa_check_03_modify")
def qa_check_03_modify(ns: dict[str, Any]) -> None:
    fn = ns.get("student_mul_probe")
    if callable(fn):
        data = _as_dict(_run(fn))
        if data is None:
            return
        _require_keys(data, ["out_data", "out_op", "parent_count"])
        _assert_close(data["out_data"], 6, "out.data")
        assert data["out_op"] == "*"
        assert data["parent_count"] == 2
        print("OK: Modify 通过。")
        return
    values = [
        ns.get("student_mul_self"),
        ns.get("student_mul_other_before_wrap"),
        ns.get("student_mul_other_after_wrap_type"),
        ns.get("student_mul_out_op"),
    ]
    if not todo_guard(values):
        return
    assert values[0] == "a"
    _assert_close(values[1], 3, "other before wrap")
    assert values[2] == "Value"
    assert values[3] == "*"
    print("OK: Modify 通过。")


@check("qa_check_source_reading_basic")
def qa_check_source_reading_basic(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_add_probe")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    _require_keys(data, ["c_data", "c_op", "c_prev_count", "d_data", "d_op", "d_prev_count"])
    _assert_close(data["c_data"], 5, "c.data")
    assert data["c_op"] == "+"
    assert data["c_prev_count"] == 2
    _assert_close(data["d_data"], 5, "d.data")
    assert data["d_op"] == "+"
    assert data["d_prev_count"] == 2
    print("OK: __add__ 保存的前向状态通过。")


@check("qa_check_source_reading_backward")
def qa_check_source_reading_backward(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_backward_probe")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    _require_keys(data, ["a_grad", "L_grad"])
    _assert_close(data["a_grad"], 2, "a.grad for L=a+a")
    _assert_close(data["L_grad"], 1, "L.grad")
    print("OK: backward 核心语义通过。")


@check("qa_check_source_debug")
def qa_check_source_debug(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_build_topo_demo")
    if fn is None:
        return
    Value = _value_class(ns)
    a = Value(2.0)
    b = Value(3.0)
    L = (a * b + a) * 2
    topo = _run(fn, L)
    if topo is None:
        return
    if not topo:
        print("请先完成 TODO：topo 还是空的。")
        return
    assert topo[-1] is L, "拓扑排序应该先父节点，最后 append 自己，所以 L 应该在最后。"
    assert a in topo and b in topo, "topo 里应该包含父节点。"
    assert len(topo) == len(set(topo)), "visited set 应该避免重复节点。"
    print("OK: topo 递归顺序和语法坑通过。")


@check("qa_check_preview_04")
def qa_check_preview_04(ns: dict[str, Any], data: Any, grad: Any) -> None:
    if not todo_guard([data, grad]):
        return
    _assert_close(data, 2.0, "data")
    _assert_close(grad, 0.0, "grad")
    print("OK: 一个可求导标量对象的最小状态通过。")


@check("qa_check_class_04_predict")
def qa_check_class_04_predict(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_minivalue_state")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    _require_keys(data, ["data", "grad", "_prev", "_op", "_backward"])
    _assert_close(data["data"], 2.0, "data")
    _assert_close(data["grad"], 0.0, "grad")
    assert callable(data["_backward"]), "_backward 应该是可调用函数。"
    print("OK: Value 最小状态通过。")


@check("qa_check_class_04_modify")
def qa_check_class_04_modify(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_add_backward_rule")
    if fn is None:
        return
    for out_grad in [1.0, 2.5, -3.0]:
        result = _as_tuple(_run(fn, out_grad), 2)
        if result is None:
            return
        _assert_close(result[0], out_grad, "self contribution")
        _assert_close(result[1], out_grad, "other contribution")
    print("OK: 加法反传规则通过。")


@check("qa_check_04_modify")
def qa_check_04_modify(ns: dict[str, Any]) -> None:
    qa_check_class_04_modify(ns)


@check("qa_check_04_forward")
def qa_check_04_forward(ns: dict[str, Any], ValueClass: type) -> None:
    try:
        a = ValueClass(2.0)
        b = ValueClass(3.0)
        c = a + b
        d = a * b
    except Exception as exc:
        print("前向还不能跑：", type(exc).__name__, exc)
        return
    assert c.data == 5.0, "加法前向 data 错。"
    assert d.data == 6.0, "乘法前向 data 错。"
    assert c._op == "+", "加法节点 _op 应该是 +。"
    assert d._op == "*", "乘法节点 _op 应该是 *。"
    print("OK: 作业 1 前向建图通过。")


@check("qa_check_04_backward")
def qa_check_04_backward(ns: dict[str, Any], ValueClass: type) -> None:
    try:
        a = ValueClass(2.0)
        b = ValueClass(3.0)
        L = (a * b + a) * 2
        L.backward()
    except Exception as exc:
        print("backward 还不能跑：", type(exc).__name__, exc)
        return
    if not (near(a.grad, 8.0) and near(b.grad, 4.0)):
        print("还没过：L=2(ab+a) 期望 a.grad=8, b.grad=4；实际:", a.grad, b.grad)
        return
    x = ValueClass(3.0)
    y = x * x
    y.backward()
    assert near(x.grad, 6.0), "L=x*x 时，x 有两条路径，grad 应该是 6。"
    r = ValueClass(-2.0)
    out = r.relu()
    out.backward()
    assert out.data == 0 and r.grad == 0, "ReLU 负半轴 data=0 且不传梯度。"
    p = ValueClass(3.0)
    q = p ** 2
    q.backward()
    assert near(p.grad, 6.0), "pow 反传要乘 out.grad。"
    print("OK: 作业 2-6 backward 核心测试通过。")


@check("qa_check_04_debug")
def qa_check_04_debug(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_pow_relu_debug_values")
    if fn is None:
        return
    result = _as_tuple(_run(fn), 3)
    if result is None:
        return
    _assert_close(result[0], 30, "pow propagated grad")
    _assert_close(result[1], 3, "ReLU(3)")
    _assert_close(result[2], 1, "ReLU local grad at x=3")
    print("OK: TinyValue Debug Lab 通过。")


@check("qa_check_preview_05")
def qa_check_preview_05(ns: dict[str, Any], inputs: Any, outputs: Any, params: Any) -> None:
    if not todo_guard([inputs, outputs, params]):
        return
    for actual, expected, name in [(inputs, 3, "inputs"), (outputs, 1, "outputs"), (params, 4, "params")]:
        _assert_close(actual, expected, name)
    print("OK: Neuron(n) 的形状通过。")


@check("qa_check_class_05_predict")
def qa_check_class_05_predict(ns: dict[str, Any]) -> None:
    value = ns.get("student_neuron3")
    if not todo_guard([value]):
        return
    assert list(value) == [3, 1, 4], f"Neuron(3) 应是 [3,1,4]，got {value}"
    print("OK: Neuron(3) 形状通过。")


@check("qa_check_05_modify")
def qa_check_05_modify(ns: dict[str, Any]) -> None:
    values = [
        ns.get("student_neuron3_weight_count"),
        ns.get("student_neuron3_bias_count"),
        ns.get("student_neuron3_output_count"),
    ]
    if not todo_guard(values):
        return
    assert values == [3, 1, 1], f"Neuron(3): 3 个权重、1 个 bias、1 个输出，got {values}"
    print("OK: Modify 通过。")


@check("qa_check_class_05_modify")
def qa_check_class_05_modify(ns: dict[str, Any]) -> None:
    value = ns.get("student_param_count")
    if not todo_guard([value]):
        return
    _assert_close(value, 13, "MLP(2,[3,1]) parameter count")
    print("OK: 参数数量 Modify 通过。")


@check("qa_check_neuron_mlp")
def qa_check_neuron_mlp(ns: dict[str, Any]) -> None:
    values = [
        ns.get("student_dot"),
        ns.get("student_neuron3_shape"),
        ns.get("student_layer23"),
        ns.get("student_mlp_layers"),
        ns.get("student_mlp_param_count"),
    ]
    if not todo_guard(values):
        return
    _assert_close(values[0], 2 * 4 + (-1) * 5 + 3 * (-2), "dot")
    assert list(values[1]) == [3, 1, 4]
    assert list(values[2]) == [3, 2]
    assert list(values[3]) == [(2, 3), (3, 1)]
    _assert_close(values[4], 13, "MLP params")
    print("OK: Neuron/Layer/MLP 形状和参数数量通过。")


@check("qa_check_bias_debug")
def qa_check_bias_debug(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_layer_bias_probe")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    _require_keys(data, ["neuron_count", "bias_count", "total_params"])
    assert data["neuron_count"] == 3
    assert data["bias_count"] == 3
    assert data["total_params"] == 9
    print("OK: bias 是每个神经元一个，你用真实 Layer 数出来了。")


@check("qa_check_preview_06")
def qa_check_preview_06(ns: dict[str, Any], value: Any) -> None:
    if not todo_guard([value]):
        return
    _assert_close(value, 4.2, "updated p")
    print("OK: -grad 更新方向通过。")


@check("qa_check_class_06_predict")
def qa_check_class_06_predict(ns: dict[str, Any]) -> None:
    qa_check_preview_06(ns, ns.get("student_new_p"))


@check("qa_check_class_06_modify")
def qa_check_class_06_modify(ns: dict[str, Any]) -> None:
    value = ns.get("student_new_p_small_lr")
    if not todo_guard([value]):
        return
    _assert_close(value, 4.92, "small lr updated p")
    print("OK: learning_rate Modify 通过。")


@check("qa_check_06_modify")
def qa_check_06_modify(ns: dict[str, Any]) -> None:
    value = ns.get("student_new_p_with_small_lr")
    if not todo_guard([value]):
        return
    _assert_close(value, 4.92, "small lr updated p")
    print("OK: Modify 通过。")


@check("qa_check_training_intro")
def qa_check_training_intro(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_training_intro_probe")
    if fn is None:
        return
    result = _as_tuple(_run(fn), 2)
    if result is None:
        return
    _assert_close(result[0], 1.0, "first sample squared error")
    _assert_close(result[1], 4.2, "updated p")
    print("OK: loss 和 update 公式通过。")


@check("qa_check_train_step")
def qa_check_train_step(ns: dict[str, Any]) -> None:
    train_step = ns.get("train_step")
    loss = ns.get("loss")
    if not callable(train_step) or not callable(loss):
        print("请先实现 train_step/loss。")
        return
    Value = _value_class(ns)
    ns["w"] = Value(0.0)
    ns["b"] = Value(0.0)
    before = loss().data
    after_reported = _run(train_step, 0.1)
    if after_reported is None:
        return
    after_now = loss().data
    assert after_now < before, "一次更新后 loss 应该下降。"
    print("OK: 一步训练让 loss 下降。before=", before, "after_now=", after_now)


@check("qa_check_update_direction")
def qa_check_update_direction(ns: dict[str, Any]) -> None:
    value = ns.get("student_correct_new_w")
    if not todo_guard([value]):
        return
    _assert_close(value, 4.2, "correct new w")
    print("OK: 更新方向 Debug 通过。")


@check("qa_check_preview_07")
def qa_check_preview_07(ns: dict[str, Any], a: Any, b: Any) -> None:
    if not todo_guard([a, b]):
        return
    _assert_close(a, 0.2, "margin y=1 score=0.2")
    _assert_close(b, 2.0, "margin y=-1 score=-2")
    print("OK: y*score 会把答对方向统一成正数。")


@check("qa_check_class_07_predict")
def qa_check_class_07_predict(ns: dict[str, Any]) -> None:
    value = ns.get("student_margin")
    if not todo_guard([value]):
        return
    _assert_close(value, 2, "margin")
    print("OK: y*score 把答对方向统一成正数。")


@check("qa_check_class_07_modify")
def qa_check_class_07_modify(ns: dict[str, Any]) -> None:
    value = ns.get("student_loss_margin2")
    if not todo_guard([value]):
        return
    _assert_close(value, 0.5, "margin=2 hinge loss")
    print("OK: 安全距离 Modify 通过。")


@check("qa_check_07_hinge")
def qa_check_07_hinge(ns: dict[str, Any]) -> None:
    label_from_score = ns.get("label_from_score")
    margin = ns.get("margin")
    hinge_loss = ns.get("hinge_loss")
    if not all(callable(fn) for fn in [label_from_score, margin, hinge_loss]):
        print("请继续完成 label_from_score / margin / hinge_loss。")
        return
    Value = _value_class(ns)
    try:
        assert label_from_score(0.3) == 1
        assert label_from_score(-0.1) == -1
        _assert_close(margin(Value(2.0), 1).data, 2.0, "positive margin")
        _assert_close(margin(Value(-2.0), -1).data, 2.0, "negative label margin")
        _assert_close(hinge_loss(Value(2.0), 1).data, 0.0, "confident hinge")
        _assert_close(hinge_loss(Value(0.2), 1).data, 0.8, "near-boundary hinge")
        _assert_close(hinge_loss(Value(-1.0), 1).data, 2.0, "wrong hinge")
    except Exception as exc:
        print("请继续完成 label_from_score / margin / hinge_loss。当前:", type(exc).__name__, exc)
        return
    print("OK: 作业 1-3 分类 loss 通过。")


@check("qa_check_07_modify")
def qa_check_07_modify(ns: dict[str, Any]) -> None:
    value = ns.get("student_hinge_margin2")
    if not todo_guard([value]):
        return
    _assert_close(value, 0.5, "hinge with safety margin 2")
    print("OK: Modify 通过。")


@check("qa_check_07_debug")
def qa_check_07_debug(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_classification_debug_cases")
    if fn is None:
        return
    result = _as_tuple(_run(fn), 3)
    if result is None:
        return
    _assert_close(result[0], 0.8, "hinge y=1 score=0.2")
    _assert_close(result[1], 3, "ReLU(3)")
    _assert_close(result[2], 2, "margin y=-1 score=-2")
    print("OK: 分类 Debug Lab 通过。")


@check("qa_check_preview_08")
def qa_check_preview_08(ns: dict[str, Any], tensor: Any) -> None:
    if not todo_guard([tensor]):
        return
    assert hasattr(tensor, "requires_grad"), "请创建一个 torch Tensor，不是字符串。"
    assert bool(tensor.requires_grad), "这个 Tensor 必须 requires_grad=True。"
    _assert_close(float(tensor.item()), 2.0, "tensor value")
    print("OK: 你创建的是会记录梯度的标量 Tensor。")


@check("qa_check_class_08_predict")
def qa_check_class_08_predict(ns: dict[str, Any]) -> None:
    if not _torch_available():
        print("torch 未安装，跳过 PyTorch 行为检查。")
        return
    fn = _student_fn(ns, "student_torch_one_step")
    if fn is None:
        return
    result = _as_tuple(_run(fn), 4)
    if result is None:
        return
    for actual, expected, name in zip(result, [6.0, 1.0, -6.0, 2.6], ["pred", "loss", "grad", "updated w"]):
        _assert_close(actual, expected, name, 1e-5)
    print("OK: PyTorch forward/backward/update 跑通。")


@check("qa_check_class_08_modify")
def qa_check_class_08_modify(ns: dict[str, Any]) -> None:
    if not _torch_available():
        print("torch 未安装，跳过 PyTorch 行为检查。")
        return
    fn = _student_fn(ns, "student_grad_accumulation_demo")
    if fn is None:
        return
    result = _as_tuple(_run(fn), 3)
    if result is None:
        return
    for actual, expected, name in zip(result, [-6.0, -12.0, -6.0], ["first grad", "second grad", "after zero grad"]):
        _assert_close(actual, expected, name, 1e-5)
    print("OK: 你亲手跑出了 zero_grad 要解决的问题。")


@check("qa_check_08_mapping")
def qa_check_08_mapping(ns: dict[str, Any], fn: Callable[[], Any] | None = None) -> None:
    if not _torch_available():
        print("torch 未安装，跳过 PyTorch 行为检查。")
        return
    fn = fn or _student_fn(ns, "student_compare_micrograd_torch")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    _require_keys(data, ["micrograd_grad", "torch_grad", "micrograd_updated_w", "torch_updated_w"])
    for key, expected in [("micrograd_grad", -6.0), ("torch_grad", -6.0), ("micrograd_updated_w", 2.6), ("torch_updated_w", 2.6)]:
        _assert_close(data[key], expected, key, 1e-5)
    print("OK: micrograd 和 PyTorch 在同一个小例子上对齐了。")


@check("qa_check_08_modify")
def qa_check_08_modify(ns: dict[str, Any]) -> None:
    value = ns.get("student_w_after_lr_001")
    if not todo_guard([value]):
        return
    _assert_close(value, 2.06, "lr=0.01 updated w")
    print("OK: Modify 通过。")


@check("qa_check_08_debug")
def qa_check_08_debug(ns: dict[str, Any]) -> None:
    if not _torch_available():
        print("torch 未安装，跳过 PyTorch 行为检查。")
        return
    fn = _student_fn(ns, "student_torch_update_and_batch_demo")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    _require_keys(data, ["updated_w", "batch_shape", "batch_times_two"])
    _assert_close(data["updated_w"], 2.6, "updated w", 1e-5)
    assert tuple(data["batch_shape"]) == (3,), f"batch shape should be (3,), got {data['batch_shape']}"
    assert [float(x) for x in data["batch_times_two"]] == [2.0, 4.0, 6.0]
    print("OK: no_grad 更新和 Tensor 批量能力都跑出来了。")


@check("qa_check_preview_09")
def qa_check_preview_09(ns: dict[str, Any], fn: Callable[[], Any] | None = None) -> None:
    fn = fn or _student_fn(ns, "student_zero_grad_demo")
    if fn is None:
        return
    result = _as_tuple(_run(fn), 3)
    if result is None:
        return
    _assert_close(result[0], 2, "first grad")
    _assert_close(result[1], 4, "second grad")
    _assert_close(result[2], 2, "after zero grad")
    print("OK: 训练循环调试第一件事就是看 zero_grad。")


@check("qa_check_class_09_predict")
def qa_check_class_09_predict(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_grad_accumulation_probe")
    if fn is None:
        return
    result = _as_tuple(_run(fn), 3)
    if result is None:
        return
    for actual, expected, name in zip(result, [3, 6, 3], ["first grad", "second grad", "after zero grad"]):
        _assert_close(actual, expected, name)
    print("OK: grad 累加症状通过。")


@check("qa_check_class_09_modify")
def qa_check_class_09_modify(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_fixed_update_demo")
    if fn is None:
        return
    result = _as_tuple(_run(fn), 2)
    if result is None:
        return
    _assert_close(result[0], 5.8, "wrong direction")
    _assert_close(result[1], 4.2, "fixed direction")
    print("OK: 更新方向 Debug 通过。")


@check("qa_check_09_modify")
def qa_check_09_modify(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_compare_update_directions")
    if fn is None:
        return
    result = _as_tuple(_run(fn), 2)
    if result is None:
        return
    _assert_close(result[0], 5.8, "wrong update")
    _assert_close(result[1], 4.2, "fixed update")
    print("OK: Modify 通过。")


@check("qa_check_debug_mapping")
def qa_check_debug_mapping(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_debug_symptom_probe")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    expected = {
        "grad_twice_without_zero": 6,
        "wrong_direction_w": 5.8,
        "fixed_direction_w": 4.2,
        "dead_relu_grad": 0,
        "unchanged_delta": 0,
    }
    _require_keys(data, expected)
    for key, expected_value in expected.items():
        _assert_close(data[key], expected_value, key)
    print("OK: Debug 症状映射通过。")


@check("qa_check_fixed_update_line")
def qa_check_fixed_update_line(ns: dict[str, Any], fixed_update: Callable[[Any, float], Any] | None = None) -> None:
    fixed_update = fixed_update or ns.get("fixed_update")
    if not callable(fixed_update):
        print("请先实现 fixed_update(p, lr)。")
        return
    Value = _value_class(ns)
    p = Value(5.0)
    p.grad = 8.0
    try:
        fixed_update(p, 0.1)
    except Exception as exc:
        print("fixed_update 还不能跑：", type(exc).__name__, exc)
        return
    if p.data == 5.0:
        print("请在 fixed_update 里真正修改 p.data。")
        return
    _assert_close(p.data, 4.2, "updated p")
    print("OK: 更新方向修复通过。")


@check("qa_check_preview_10")
def qa_check_preview_10(ns: dict[str, Any], order: Any) -> None:
    if not todo_guard([order]):
        return
    assert list(order) == ["forward", "loss", "backward", "update"]
    print("OK: 项目主链路顺序通过。")


@check("qa_check_class_10_predict")
def qa_check_class_10_predict(ns: dict[str, Any]) -> None:
    qa_check_preview_10(ns, ns.get("student_order"))


@check("qa_check_class_10_modify")
def qa_check_class_10_modify(ns: dict[str, Any]) -> None:
    value = ns.get("student_param_count")
    if not todo_guard([value]):
        return
    _assert_close(value, 13, "MLP params")
    print("OK: 改隐藏层参数数量通过。")


@check("qa_check_10_modify")
def qa_check_10_modify(ns: dict[str, Any]) -> None:
    value = ns.get("student_param_count_mlp_2_3_1")
    if not todo_guard([value]):
        return
    _assert_close(value, 13, "MLP params")
    print("OK: Modify 通过。")


@check("qa_check_10_project_functions")
def qa_check_10_project_functions(ns: dict[str, Any]) -> None:
    try:
        s = ns["predict_score"](ns["xs"][0])
        label = ns["predict_label"](ns["xs"][0])
        L = ns["total_loss"]()
        acc = ns["accuracy"]()
    except Exception as exc:
        print("请继续实现项目函数。当前:", type(exc).__name__, exc)
        return
    if _contains_todo([s, label, L, acc]):
        print("请继续实现 TODO：有函数返回 None。")
        return
    assert hasattr(s, "data"), "predict_score 应该返回 Value。"
    assert label in [-1, 1]
    assert hasattr(L, "data"), "total_loss 应该返回 Value。"
    assert 0 <= acc <= 1
    print("OK: 项目函数骨架通过。loss=", round(L.data, 4), "acc=", round(acc, 3))


@check("qa_check_10_train")
def qa_check_10_train(ns: dict[str, Any]) -> None:
    train = ns.get("train")
    if not callable(train):
        print("请先实现 train。")
        return
    try:
        history = train(steps=20, learning_rate=0.05)
    except Exception as exc:
        print("请继续实现 train。当前:", type(exc).__name__, exc)
        return
    if not history:
        print("train 应该返回 loss history。")
        return
    assert history[-1] <= history[0], "训练后 loss 应该不升高。"
    print("OK: 训练循环通过。first/last=", round(history[0], 4), round(history[-1], 4))


@check("qa_check_10_debug")
def qa_check_10_debug(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_project_debug_probe")
    if fn is None:
        return
    result = _as_tuple(_run(fn), 3)
    if result is None:
        return
    assert result[0] is True, "loss 下降但 accuracy 暂时不变是可能的。"
    _assert_close(result[1], 0, "unchanged param delta")
    _assert_close(result[2], 0, "grad after zero")
    print("OK: Mini Project Debug Lab 通过。")


@check("qa_check_preview_11")
def qa_check_preview_11(ns: dict[str, Any], probe: Any, nn: Any = None) -> None:
    if callable(probe):
        data = _as_dict(_run(probe))
        if data is None:
            return
        _require_keys(data, ["engine_has_value", "nn_has_mlp"])
        assert data["engine_has_value"] is True
        assert data["nn_has_mlp"] is True
        print("OK: 项目结构入口通过。")
        return
    if not todo_guard([probe, nn]):
        return
    assert probe in ["Value", "Value/autograd"]
    assert nn in ["MLP", "Neuron/Layer/MLP"]
    print("OK: 项目结构入口通过。")


@check("qa_check_class_11_predict")
def qa_check_class_11_predict(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_structure_probe")
    if fn is not None:
        data = _as_dict(_run(fn))
        if data is None:
            return
        _require_keys(data, ["engine_has_value", "nn_has_neuron", "nn_has_mlp"])
        assert data["engine_has_value"] is True
        assert data["nn_has_neuron"] is True
        assert data["nn_has_mlp"] is True
        print("OK: 文件职责预测通过。")
        return
    engine = ns.get("student_engine_role")
    nn = ns.get("student_nn_role")
    if not todo_guard([engine, nn]):
        return
    assert "Value" in engine
    assert "Neuron" in nn or "MLP" in nn
    print("OK: 文件职责预测通过。")


@check("qa_check_class_11_modify")
def qa_check_class_11_modify(ns: dict[str, Any]) -> None:
    if not _torch_available():
        print("torch 未安装，跳过 PyTorch 行为检查。")
        return
    fn = _student_fn(ns, "student_pytorch_gate_demo", "student_update_bridge_demo")
    if fn is None:
        return
    result = _run(fn)
    if result is None:
        return
    if isinstance(result, Mapping):
        data = dict(result)
        values = [data.get("loss"), data.get("grad"), data.get("updated_w")]
    else:
        values = list(result) if isinstance(result, (list, tuple)) else None
    if values is None:
        print("请返回 loss/grad/updated_w，或返回两个更新后的 w。")
        return
    if len(values) == 3:
        _assert_close(values[0], 1.0, "loss", 1e-5)
        _assert_close(values[1], -6.0, "grad", 1e-5)
        _assert_close(values[2], 2.6, "updated w", 1e-5)
    elif len(values) == 2:
        _assert_close(values[0], 2.6, "manual updated w", 1e-5)
        _assert_close(values[1], 2.6, "optimizer updated w", 1e-5)
    else:
        raise AssertionError(f"返回长度不对: {values}")
    print("OK: 下一阶段入口不是口号，而是能跑的小实验。")


@check("qa_check_11_file_roles")
def qa_check_11_file_roles(ns: dict[str, Any], fn: Callable[[], Any] | None = None) -> None:
    fn = fn or _student_fn(ns, "student_inspect_micrograd_files")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    _require_keys(data, ["engine_has_value", "nn_has_mlp", "demo_exists"])
    assert data["engine_has_value"] is True
    assert data["nn_has_mlp"] is True
    assert data["demo_exists"] is True
    print("OK: 你通过项目结构定位了 engine/nn/demo。")


@check("qa_check_11_torch_mapping")
def qa_check_11_torch_mapping(ns: dict[str, Any], fn: Callable[[], Any] | None = None) -> None:
    if not _torch_available():
        print("torch 未安装，跳过 PyTorch 行为检查。")
        return
    fn = fn or _student_fn(ns, "student_torch_bridge_probe")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    _require_keys(data, ["micrograd_grad", "torch_grad", "micrograd_updated_w", "torch_updated_w"])
    for key, expected in [("micrograd_grad", -6), ("torch_grad", -6), ("micrograd_updated_w", 2.6), ("torch_updated_w", 2.6)]:
        _assert_close(data[key], expected, key, 1e-5)
    print("OK: PyTorch 映射通过。")


@check("qa_check_11_modify")
def qa_check_11_modify(ns: dict[str, Any]) -> None:
    qa_check_class_11_modify(ns)


@check("qa_check_11_next_route")
def qa_check_11_next_route(ns: dict[str, Any], route: Any) -> None:
    if not todo_guard([route]):
        return
    assert isinstance(route, list) and len(route) >= 5, "路线至少包含 5 个阶段。"
    stages = [item.get("stage") for item in route if isinstance(item, Mapping)]
    for required in ["PyTorch", "D2L", "Happy-LLM", "LLMs-from-scratch", "nanoGPT"]:
        assert required in stages, f"缺少阶段：{required}"
    for item in route:
        assert item.get("pass_question"), "每个阶段都要有过关问题，不要只列资源名。"
    print("OK: 下一阶段路线通过。")


@check("qa_check_11_debug")
def qa_check_11_debug(ns: dict[str, Any]) -> None:
    fn = _student_fn(ns, "student_route_debug")
    if fn is None:
        return
    data = _as_dict(_run(fn))
    if data is None:
        return
    _require_keys(data, ["import_only_passes", "too_many_resources", "has_pass_questions"])
    assert data["import_only_passes"] is False
    assert data["too_many_resources"] is True
    assert data["has_pass_questions"] is False
    print("OK: 路线 Debug Lab 通过。")


def qa_check(name: str, namespace: dict[str, Any], *args: Any) -> None:
    """Run a registered feedback check inside a notebook namespace."""
    if name not in CHECKS:
        raise KeyError(f"Unknown qa_check: {name}")
    return CHECKS[name](namespace, *args)
