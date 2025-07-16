from absl.testing.absltest import TestCase, main  # type: ignore

from fusebuild.core.graph import sort_graph


class TestExample1(TestCase):
    def test_sort_graph(self):
        graph = {}
        graph[0] = []
        graph[1] = [0]
        graph[2] = [1]
        graph[3] = [0, 1]
        graph[4] = [3]

        sorted = sort_graph(graph)
        print(sorted)

        self.assertEqual(sorted.index(0), 0, "Everything depends on 0, must be first")
        self.assertTrue(sorted.index(1) < sorted.index(2), "2 depends on 1")
        self.assertTrue(sorted.index(1) < sorted.index(3), "3 depends on 1")
        self.assertTrue(
            sorted.index(1) < sorted.index(4), "4 depends on 3 depends on 1"
        )
        self.assertTrue(sorted.index(3) < sorted.index(4), "4 depends on 3")

    def test_sort_graph_cyclic(self):
        graph = {}
        graph[0] = [4]
        graph[1] = [0]
        graph[2] = [1]
        graph[3] = [0, 1]
        graph[4] = [3]

        sorted = sort_graph(graph)
        print(sorted)
        # All we can test is that all 5 elements are returned - the order isn't important
        self.assertEqual(len(sorted), 5)
        self.assertEqual(len(set(sorted)), 5)


if __name__ == "__main__":
    main()
