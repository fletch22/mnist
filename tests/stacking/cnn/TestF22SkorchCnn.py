from unittest import TestCase

from app.stacking.cnn import F22SkorchCnn


class TestF22SkorchCnn(TestCase):

    def test_try_it(self):
        # Arrange
        # Act
        F22SkorchCnn.try_it()