import os
from unittest import TestCase
from app.config import config

import logging


class TestMnistQuick(TestCase):

    def test_stitch(self):
        stacking_path = os.path.join(config.PROJECT_ROOT_PATH, 'app', 'stacking')
        file1 = os.path.join(stacking_path, 'cnn', 'F22StackingCnn.py')
        file2 = os.path.join(stacking_path, 'mnist_quick.py')

        filenames = [file1, file2]

        output_path = os.path.join(config.SESSION_DIR, 'mnist-output.py')
        with open(output_path, 'w') as outfile:
            for fname in filenames:
                with open(fname) as infile:
                    outfile.write(infile.read())

        logging.info(output_path)
