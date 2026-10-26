import unittest
from train_jsplib import LabelTable


class SourceLabelTests(unittest.TestCase):
    def test_labels_are_read_from_dedicated_source_column(self):
        parser = LabelTable()
        parser.feed('<tr><td>instance1</td><td>10 x 10</td><td>random</td><td>5</td><td>8</td><td style="color:red">open</td><td>no proof</td></tr>')
        self.assertEqual(parser.labels, {"instance1": "open"})
        parser.feed('<tr><td>instance2</td><td>10 x 10</td><td>random</td><td>5</td><td>5</td><td>closed</td><td>bounds match</td></tr>')
        self.assertEqual(parser.labels["instance2"], "closed")
