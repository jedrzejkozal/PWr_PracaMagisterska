import pytest

from Models.HilbertLayer.CommandsGenerator import *


class TestCommandsGenerator(object):

    @pytest.fixture
    def sut(self):
        return CommandsGenerator()


    def test_generate_commands_for_deegre_1_generates_A(self, sut):
        result = sut.generate_commands(1)
        assert result == ["left", "F", "right", "F", "right", "F", "left"]


    def test_generate_commands_for_deegre_2_generates_right_sequence(self, sut):
        result = sut.generate_commands(2)
        expected = ["left", "right", "F", "left", "F", "left", "F", "right", "F", "right",
                    "left", "F", "right", "F", "right", "F", "left", "F",
                    "left", "F", "right", "F", "right", "F", "left", "right", "F",
                    "right", "F", "left", "F", "left", "F", "right", "left"]

        assert result == expected
