import unittest

import numpy as np

from mini_pytorch import (
    Device,
    DispatchError,
    DispatchKey,
    Tensor,
    backend_for,
    dispatcher,
    functional,
    list_backends,
    nn,
    no_grad,
)
from mini_pytorch.storage import Storage


class TestDevice(unittest.TestCase):
    def test_parse_equality_hash_and_validation(self):
        self.assertEqual(Device("CPU"), Device("cpu"))
        self.assertEqual(str(Device("edu:0")), "edu:0")
        self.assertEqual(len({Device("cpu"), Device("cpu"), Device("edu")}), 2)
        with self.assertRaises(ValueError):
            Device("cuda")
        with self.assertRaises(ValueError):
            Device("edu:-1")
        with self.assertRaises(ValueError):
            Device("cpu:x")

    def test_backend_registry_and_storage_identity(self):
        self.assertEqual(list_backends(), ("cpu", "edu"))
        self.assertEqual(backend_for("cpu").device_type, "cpu")
        self.assertEqual(backend_for("edu").device_type, "edu")

        storage = Storage([1.0, 2.0], device="edu")
        self.assertEqual(storage.device, Device("edu"))
        self.assertEqual(storage.backend.device_type, "edu")
        np.testing.assert_allclose(storage.numpy(), [1.0, 2.0])

        empty = Storage.empty(3, np.float32, device="edu")
        self.assertEqual(empty.size, 3)
        self.assertEqual(empty.dtype, np.dtype(np.float32))
        self.assertEqual(empty.device, Device("edu"))


class TestTensorTransfer(unittest.TestCase):
    def test_tensor_exposes_device(self):
        cpu = Tensor([1, 2])
        edu = Tensor([1, 2], device="edu")
        self.assertEqual(cpu.device, Device("cpu"))
        self.assertEqual(cpu.storage.device, Device("cpu"))
        self.assertEqual(edu.device, Device("edu"))
        self.assertIn("device=edu", repr(edu))

    def test_same_device_noop_and_copy(self):
        x = Tensor(np.arange(6).reshape(2, 3))
        self.assertIs(x.to("cpu"), x)

        copied = x.to("cpu", copy=True)
        self.assertIsNot(copied, x)
        self.assertIsNot(copied.storage, x.storage)
        np.testing.assert_array_equal(copied.data, x.data)

    def test_cross_device_transfer_breaks_aliasing(self):
        x = Tensor(np.arange(6.0).reshape(2, 3))
        view = x.T
        moved = view.to("edu")
        self.assertEqual(moved.device, Device("edu"))
        self.assertIsNot(moved.storage, x.storage)
        self.assertTrue(moved.is_contiguous())
        np.testing.assert_array_equal(moved.data, view.data)

        x.add_(10.0)
        np.testing.assert_array_equal(moved.data, np.array([[0, 3], [1, 4], [2, 5]], dtype=float))

    def test_transfer_is_differentiable_identity(self):
        x = Tensor([2.0, -3.0], requires_grad=True)
        y = x.to("edu")
        self.assertEqual(y.device, Device("edu"))
        self.assertTrue(y.requires_grad)
        self.assertEqual(y.grad_fn, "_ToDeviceBackward")

        loss = (y * y).sum()
        loss.backward()
        np.testing.assert_allclose(x.grad, [4.0, -6.0])

    def test_direct_function_ops_preserve_device(self):
        x = Tensor([0.0, 1.0], device="edu", requires_grad=True)
        y = x.exp()
        self.assertEqual(y.device, Device("edu"))
        y.sum().backward()
        np.testing.assert_allclose(x.grad, np.exp([0.0, 1.0]))


class TestDeviceAwareDispatch(unittest.TestCase):
    def test_cpu_and_edu_select_different_backend_keys(self):
        cpu_a = Tensor([1.0])
        cpu_b = Tensor([2.0])
        edu_a = Tensor([1.0], device="edu")
        edu_b = Tensor([2.0], device="edu")

        self.assertEqual(
            dispatcher.explain("aten::add.Tensor", cpu_a, cpu_b).selected_key,
            DispatchKey.CPU,
        )
        self.assertEqual(
            dispatcher.explain("aten::add.Tensor", edu_a, edu_b).selected_key,
            DispatchKey.EDU,
        )
        keys = dispatcher.registered_keys("aten::add.Tensor")
        self.assertIn(DispatchKey.CPU, keys)
        self.assertIn(DispatchKey.EDU, keys)

    def test_autograd_and_no_grad_preserve_backend_key(self):
        a = Tensor([1.0], device="edu", requires_grad=True)
        b = Tensor([2.0], device="edu", requires_grad=True)
        decision = dispatcher.explain("aten::add.Tensor", a, b)
        self.assertEqual(decision.selected_key, DispatchKey.AUTOGRAD)
        out = a + b
        self.assertEqual(out.device, Device("edu"))

        with no_grad():
            decision = dispatcher.explain("aten::add.Tensor", a, b)
            self.assertEqual(decision.selected_key, DispatchKey.NOGRAD)
            out = a + b
        self.assertEqual(out.device, Device("edu"))
        self.assertFalse(out.requires_grad)

    def test_mixed_device_fails_before_backend_execution(self):
        cpu = Tensor([1.0])
        edu = Tensor([2.0], device="edu")
        with self.assertRaisesRegex(DispatchError, "different devices"):
            _ = cpu + edu
        with self.assertRaisesRegex(ValueError, "different devices"):
            _ = cpu - edu

    def test_scalar_promotion_stays_on_owning_device(self):
        x = Tensor([1.0, 2.0], device="edu")
        y = x + 3.0
        z = 2.0 * x
        self.assertEqual(y.device, Device("edu"))
        self.assertEqual(z.device, Device("edu"))
        np.testing.assert_allclose(y.data, [4.0, 5.0])
        np.testing.assert_allclose(z.data, [2.0, 4.0])


class TestModuleAndNNBackend(unittest.TestCase):
    def test_module_to_moves_parameters_and_buffers(self):
        model = nn.Sequential(
            nn.Conv2d(1, 2, 3, padding=1),
            nn.BatchNorm2d(2),
        )
        parameter_ids = [id(p) for p in model.parameters()]
        model.to("edu")
        self.assertTrue(all(p.device == Device("edu") for p in model.parameters()))
        self.assertTrue(all(b.device == Device("edu") for _, b in model.named_buffers()))
        self.assertEqual(parameter_ids, [id(p) for p in model.parameters()])
        self.assertTrue(all(p.is_leaf for p in model.parameters()))

    def test_conv2d_and_pool_route_on_edu(self):
        x = Tensor(np.arange(16.0).reshape(1, 1, 4, 4), device="edu", requires_grad=True)
        weight = Tensor(np.ones((1, 1, 3, 3)), device="edu", requires_grad=True)
        out = functional.conv2d(x, weight, padding=1, implementation="im2col")
        self.assertEqual(out.device, Device("edu"))
        pooled = functional.max_pool2d(out, 2)
        self.assertEqual(pooled.device, Device("edu"))
        pooled.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)


if __name__ == "__main__":
    unittest.main()
