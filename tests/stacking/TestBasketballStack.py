from unittest import TestCase

from app.stacking import basketball_stack


class TestBasketballStack(TestCase):

    def test_stack(self):
        basketball_stack.main()