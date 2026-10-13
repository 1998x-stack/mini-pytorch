import unittest

from mini_pytorch import Tensor
from mini_pytorch.compiler import analyze_graph, capture


class TestCaptureIdentityLifetime(unittest.TestCase):
    def test_short_lived_temporaries_keep_distinct_graph_values(self):
        def fn(x):
            # Each temporary is deliberately deleted before the next allocation.
            # CPython is therefore free to recycle object ids. Capture must track
            # Tensor *object identity* for the whole session rather than trusting
            # a bare id(tensor) after the object has died.
            for _ in range(64):
                temporary = x + x
                del temporary
            return x * x

        captured = capture(fn, Tensor([1.0, 2.0]))
        output_ids = [node.outputs.value_id for node in captured.graph.nodes]

        self.assertEqual(len(output_ids), 65)
        self.assertEqual(len(set(output_ids)), len(output_ids))
        # Producer analysis is the invariant that previously exposed id reuse.
        analyze_graph(captured.graph)


if __name__ == "__main__":
    unittest.main()
