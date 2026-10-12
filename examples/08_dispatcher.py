"""mini-pytorch v0.5: inspect CPU / Autograd / NoGrad dispatch paths."""
from mini_pytorch import DispatchKey, Tensor, dispatcher, no_grad


def show(label, decision):
    print(f"\n[{label}]")
    print("operator      :", decision.operator)
    print("keyset        :", decision.keyset)
    print("selected key  :", decision.selected_key.value)
    print("implementation:", decision.implementation)


def main():
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])

    cpu = dispatcher.explain("aten::add.Tensor", a, b)
    show("ordinary tensors", cpu)
    print("a + b =", (a + b).data)

    x = Tensor([2.0, 3.0], requires_grad=True)
    autograd = dispatcher.explain("aten::mul.Tensor", x, x)
    show("requires_grad", autograd)

    loss = ((x * x) + x).sum()
    print("loss grad_fn:", loss.grad_fn)
    loss.backward()
    print("x.grad       :", x.grad)

    with no_grad():
        nograd = dispatcher.explain("aten::add.Tensor", x, x)
        show("inside no_grad", nograd)
        y = x + x
        print("result requires_grad:", y.requires_grad)
        print("result grad_fn      :", y.grad_fn)

    print("\n[registry]")
    for operator in dispatcher.list_operators():
        keys = ", ".join(key.value for key in dispatcher.registered_keys(operator))
        print(f"{operator:24s} -> {keys}")

    print("\n[important mental model]")
    print(
        "Autograd and NoGrad change routing policy; the CPU numerical kernel "
        "does not need to know about dynamic-graph construction."
    )


if __name__ == "__main__":
    main()
