from unittest import TestCase

from app.util.object_utils import fix_illegal_keys


class TestObjectUtils(TestCase):

    def test_replace_illegal_plain(self):
        # Arrange
        dict_thingy = {"foo": {"thing": "bar"}}

        # Act
        result_thingy = fix_illegal_keys(dict_thingy)

        # Assert
        assert dict_thingy == result_thingy

    def test_replace_illegal_illegal_char(self):
        # Arrange
        dict_thingy = {"foo%": {"t/hi+n-g": "bar"}}

        # Act
        result_thingy = fix_illegal_keys(dict_thingy)

        # Assert
        assert {'foo_pct': {'t_hiplusnminusg': 'bar'}} == result_thingy