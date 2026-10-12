import unittest

import numpy as np

from mini_pytorch import Tensor, no_grad
from mini_pytorch.dispatcher import (
    DispatchError,
    DispatchKey,
    DispatchKeySet,
    Dispatcher,
    OperatorSchema,
    dispatcher,
)


class TestDispatcherRegistry(unittest.TestCase):
    def test_define_and_inspect_schema(self):
        d = Dispatcher()
        schema = OperatorSchema(
            "demo",
            "scale",
            "Tensor",
            "(Tensor self, float factor) -> Tensor",
        )
        d.define(schema)
        self.assertEqual(schema.qualified_name, "demo::scale.Tensor")
        self.assertEqual(d.schema("demo::scale.Tensor"), schema)
        self.assertEqual(d.list_operators(), ("demo::scale.Tensor",))

    def test_duplicate_schema_fails(self):
        d = Dispatcher()
        schema = OperatorSchema("demo", "op")
        d.define(schema)
        with self.assertRaisesRegex(DispatchError, "already defined"):
            d.define(schema)

    def test_kernel_for_undefined_operator_fails(self):
        d = Dispatcher()
        with self.assertRaisesRegex(DispatchError, "undefined operator"):
            d.register("demo::missing.default", DispatchKey.CPU, lambda x: x)

    def test_duplicate_kernel_fails(self):
        d = Dispatcher()
        qname = d.define(OperatorSchema("demo", "op")).qualified_name
        d.register(qname, DispatchKey.CPU, lambda x: x)
        with self.assertRaisesRegex(DispatchError, "already registered"):
            d.register(qname, DispatchKey.CPU, lambda x: x)

    def test_unknown_operator_fails(self):
        d = Dispatcher()
        with self.assertRaisesRegex(DispatchError, "unknown operator"):
            d.dispatch("demo::missing.default", Tensor(1.0))

    def test_explicit_unsupported_high_priority_key_fails(self):
        x = Tensor([1.0])
        with self.assertRaisesRegex(DispatchError, "CUDA"):
            dispatcher.redispatch(
                "aten::add.Tensor",
                DispatchKeySet([DispatchKey.CUDA, DispatchKey.CPU]),
                x,
                x,
            )


class TestBuiltinDispatch(unittest.TestCase):
    def test_builtin_registry_contains_core_ops(self):
        names = dispatcher.list_operators()
        self.assertIn("aten::add.Tensor", names)
        self.assertIn("aten::mul.Tensor", names)
        self.assertIn("aten::matmul.default", names)
        self.assertIn("aten::relu.default", names)
        self.assertIn("aten::sum.default", names)
        self.assertEqual(
            dispatcher.registered_keys("aten::add.Tensor"),
            (DispatchKey.AUTOGRAD, DispatchKey.CPU),
        )

    def test_non_grad_tensor_selects_cpu(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        decision = dispatcher.explain("aten::add.Tensor", a, b)
        self.assertEqual(decision.selected_key, DispatchKey.CPU)
        out = a + b
        self.assertFalse(out.requires_grad)
        np.testing.assert_allclose(out.data, [4.0, 6.0])

    def test_requires_grad_selects_autograd_before_cpu(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0])
        decision = dispatcher.explain("aten::add.Tensor", a, b)
        self.assertEqual(decision.selected_key, DispatchKey.AUTOGRAD)
        out = a + b
        self.assertTrue(out.requires_grad)
        self.assertEqual(out.grad_fn, "AddBackward")

    def test_no_grad_selects_fallback_and_redispatches(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        with no_grad():
            decision = dispatcher.explain("aten::add.Tensor", a, a)
            self.assertEqual(decision.selected_key, DispatchKey.NOGRAD)
            self.assertEqual(decision.implementation, "fallback")
            out = a + a
        self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)
        np.testing.assert_allclose(out.data, [2.0, 4.0])

    def test_explicit_redispatch_reaches_cpu(self):
        a = Tensor([2.0])
        b = Tensor([5.0])
        out = dispatcher.redispatch(
            "aten::mul.Tensor",
            DispatchKeySet([DispatchKey.CPU]),
            a,
            b,
        )
        self.assertFalse(out.requires_grad)
        np.testing.assert_allclose(out.data, [10.0])

    def test_dispatched_add_and_mul_backward(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = ((x * x) + x).sum()
        y.backward()
        np.testing.assert_allclose(x.grad, [5.0, 7.0])

    def test_dispatched_relu_backward(self):
        x = Tensor([-2.0, 0.5, 3.0], requires_grad=True)
        x.relu().sum().backward()
        np.testing.assert_allclose(x.grad, [0.0, 1.0, 1.0])

    def test_dispatched_matmul_backward(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[2.0, 0.0], [1.0, 2.0]], requires_grad=True)
        (a @ b).sum().backward()
        np.testing.assert_allclose(a.grad, [[2.0, 3.0], [2.0, 3.0]])
        np.testing.assert_allclose(b.grad, [[4.0, 4.0], [6.0, 6.0]])

    def test_saved_version_check_survives_dispatch(self):
        x = Tensor([2.0, 3.0], requires_grad=True)
        loss = (x * x).sum()
        x.detach().add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            loss.backward()


if __name__ == "__main__":
    unittest.main()
