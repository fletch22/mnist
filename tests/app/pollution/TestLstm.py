from unittest import TestCase

from app.pollution import lstm
from tests.app.pollution.TestLstmTest import main


class TestLstm(TestCase):

    def test_show_pollution(self):
        # Arrange
        # Act
        lstm.show_pollution()

    def test_pollution_prep(self):
        # Arrange
        # Act
        lstm.prep_pollution_data()

    def test_main(self):
        # Arrange
        # Act
        lstm.main()

        # main()