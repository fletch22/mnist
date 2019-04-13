from unittest import TestCase

from app import naive_bayes

class TestBayes(TestCase):

    def test_naive_bayes_main(self):
        # Arrange
        # Act
        naive_bayes.main()

        # Assert
