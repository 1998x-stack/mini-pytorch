import unittest

import numpy as np

from mini_pytorch import Tensor, dispatcher
from mini_pytorch.compiler import GraphBreakError, GraphInterpreter, capture


class TestGraphIRCapture(unittest.TestCase):
    def test_capture_records_inputs_nodes_and_output(self):
        x = Tensor([1.0, -2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)

        captured = capture(lambda a, b: ((a * b) + a).relu().sum(), x, y)
        graph = captured.graph

        self.assertEqual(len(graph.inputs), 2)
        self.assertEqual(
            [node.operator for node in graph.nodes],
            [
                "aten::mul.Tensor",
                "aten::add.Tensor",
                "aten::relu.default",
                "aten::sum.default",
            ],
        )
        self.assertIn("aten::mul.Tensor", graph.format())
        self.assertIn("return %v", graph.format())

    def test_ir_text_is_deterministic(self):
        def fn(x):
            return ((x * 2.0) + x).sum(axis=0, keepdims=True)

        a = capture(fn, Tensor([1.0, 2.0], requires_grad=True)).graph.format()
        b = capture(fn, Tensor([9.0, 8.0], requires_grad=True)).graph.format()
        # The scalar constant is identical; input values themselves are not
        # embedded into graph input metadata, so the textual IR is stable.
        self.assertEqual(a, b)

    def test_tensor_leaf_constant_is_captured_by_value(self):
        weight = Tensor([2.0, 3.0], requires_grad=True)
        x = Tensor([4.0, 5.0], requires_grad=True)
        graph = capture(lambda a: (a * weight).sum(), x).graph
        constants = [v for v in graph.values.values() if v.kind == "constant"]
        self.assertEqual(len(constants), 1)
        np.testing.assert_allclose(constants[0].constant.data, weight.data)
        self.assertTrue(constants[0].constant.requires_grad)

    def test_kwargs_are_preserved(self):
        x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        graph = capture(lambda a: a.sum(axis=1, keepdims=True), x).graph
        node = graph.nodes[0]
        self.assertEqual(node.kwargs, {"axis": 1, "keepdims": True})

    def test_nested_capture_is_rejected(self):
        x = Tensor([1.0])

        def outer(a):
            capture(lambda b: b + b, a)
            return a + a

        with self.assertRaisesRegex(RuntimeError, "nested"):
            capture(outer, x)

    def test_duplicate_input_object_is_rejected(self):
        x = Tensor([1.0])
        with self.assertRaisesRegex(ValueError, "same Tensor"):
            capture(lambda a, b: a + b, x, x)

    def test_unsupported_direct_op_causes_graph_break(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaisesRegex(GraphBreakError, "unsupported"):
            capture(lambda a: a.exp() + a, x)

    def test_unsupported_direct_output_causes_graph_break(self):
        x = Tensor([1.0], requires_grad=True)
        with self.assertRaises(GraphBreakError):
            capture(lambda a: a.exp(), x)


class TestGraphInterpreter(unittest.TestCase):
    @staticmethod
    def _fn(x, y):
        return (((x * y) + x).relu()).sum()

    def test_forward_replay_matches_eager(self):
        tx = Tensor([1.5, -2.0], requires_grad=True)
        ty = Tensor([2.0, 3.0], requires_grad=True)
        graph = capture(self._fn, tx, ty).graph

        x = Tensor([4.0, -1.0], requires_grad=True)
        y = Tensor([0.5, 5.0], requires_grad=True)
        eager = self._fn(x, y)

        rx = Tensor([4.0, -1.0], requires_grad=True)
        ry = Tensor([0.5, 5.0], requires_grad=True)
        replay = GraphInterpreter(graph).run(rx, ry)

        np.testing.assert_allclose(replay.data, eager.data)

    def test_backward_input_gradients_match_eager(self):
        graph = capture(
            self._fn,
            Tensor([1.0, 2.0], requires_grad=True),
            Tensor([3.0, 4.0], requires_grad=True),
        ).graph

        x = Tensor([2.0, -1.0], requires_grad=True)
        y = Tensor([4.0, 5.0], requires_grad=True)
        eager = self._fn(x, y)
        eager.backward()

        rx = Tensor([2.0, -1.0], requires_grad=True)
        ry = Tensor([4.0, 5.0], requires_grad=True)
        replay = GraphInterpreter(graph)(rx, ry)
        replay.backward()

        np.testing.assert_allclose(rx.grad, x.grad)
        np.testing.assert_allclose(ry.grad, y.grad)

    def test_replay_on_edu_uses_normal_dispatcher(self):
        graph = capture(
            lambda x: (x * x).sum(),
            Tensor([1.0, 2.0], requires_grad=True, device="edu"),
        ).graph
        x = Tensor([3.0, 4.0], requires_grad=True, device="edu")
        out = GraphInterpreter(graph)(x)
        self.assertEqual(str(out.device), "edu")
        out.backward()
        np.testing.assert_allclose(x.grad, [6.0, 8.0])

    def test_wrong_input_count_fails(self):
        graph = capture(lambda x: x + x, Tensor([1.0])).graph
        with self.assertRaisesRegex(ValueError, "expects 1 inputs"):
            GraphInterpreter(graph)()

    def test_replay_calls_dispatcher(self):
        graph = capture(lambda x: (x + x).sum(), Tensor([1.0])).graph
        calls = []
        original = dispatcher.dispatch

        def spy(operator, *args, **kwargs):
            calls.append(operator)
            return original(operator, *args, **kwargs)

        dispatcher.dispatch = spy
        try:
            GraphInterpreter(graph)(Tensor([2.0]))
        finally:
            dispatcher.dispatch = original

        self.assertEqual(calls, ["aten::add.Tensor", "aten::sum.default"])

    def test_captured_tensor_constant_replays_without_alias(self):
        c = Tensor([2.0, 3.0])
        graph = capture(lambda x: x * c, Tensor([1.0, 1.0])).graph
        c.data[...] = 99.0
        out = GraphInterpreter(graph)(Tensor([4.0, 5.0]))
        np.testing.assert_allclose(out.data, [8.0, 15.0])


if __name__ == "__main__":
    unittest.main()
