'''batch skip-gram unit_test'''
import unittest
from .data import SkipGramDataSet
import os
'''set absolute path'''
_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_FILE = os.path.join(_CURRENT_DIR, "./data/test.txt")
BATCH_SIZE = 16

class TestDataSet(unittest.TestCase):

  '''test wwo batches'''
  def testGenBatchInputs(self):
    ds = SkipGramDataSet(file=TEST_FILE)
    features, labels = ds.gen_batch_inputs(BATCH_SIZE, 1)
    for i in range(BATCH_SIZE):
      print("%s --> %s" % (ds.id2word[features[i]], ds.id2word[labels[i]]))
    for i in range(16):
      features, labels = ds.gen_batch_inputs(BATCH_SIZE, 1)
      for i in range(BATCH_SIZE):
        print("%s --> %s" % (ds.id2word[features[i]], ds.id2word[labels[i]]))

if __name__ == "__main__":
  unittest.main()
