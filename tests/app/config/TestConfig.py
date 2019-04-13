import logging
import os
from unittest import TestCase

from app.config import config
from app.services import file_services


class TestConfig(TestCase):

    def test_clean(self):
        # Arrange
        # Act
        config.clean_temp_folders

