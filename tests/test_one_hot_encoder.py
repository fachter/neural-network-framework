import unittest
from netneural.network.one_hot_encoder import OneHotEncoder
import numpy as np


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_given_new_instance_then_encode_new():
        encoder = OneHotEncoder()
        y = ["rotate", "idle", "rotate", "swipe", "idle", "rotate"]
        expected_encoding = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ])

        result = encoder.encode(y)

        np.testing.assert_array_equal(result, expected_encoding)

    @staticmethod
    def test_given_existing_instance_then_save_and_use_same_encoding():
        encoder = OneHotEncoder()
        first_y = ["rotate", "idle", "swipe"]
        second_y = ["rotate", "rotate", "rotate", "swipe"]
        expected_encoding = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        encoder.encode(first_y)
        result = encoder.encode(second_y)

        np.testing.assert_array_equal(result, expected_encoding)

    @staticmethod
    def test_given_encoding_then_decode_to_words():
        encoder = OneHotEncoder()
        y = ["rotate", "idle", "swipe"]
        expected_labels = ["swipe"]

        encoded = encoder.encode(y)
        original_labels = encoder.decode(encoded)
        new_label = encoder.decode(np.array([[0, 0, 1]]))

        np.testing.assert_array_equal(y, original_labels)
        np.testing.assert_array_equal(expected_labels, new_label)

    @staticmethod
    def test_given_unique_labels_then_instantiate_same_encoder():
        saved_unique_labels = np.array(["idle", "rotate", "swipe"])
        encoder = OneHotEncoder(saved_unique_labels)
        y = ["rotate", "rotate", "rotate", "swipe"]
        expected_encoding = np.array([
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ])

        result = encoder.encode(y)

        np.testing.assert_array_equal(result, expected_encoding)

    @staticmethod
    def test_given_unique_labels_then_instantiate_same_decoder():
        saved_unique_labels = np.array(["idle", "rotate", "swipe_left", "swipe_right"])
        encoder = OneHotEncoder(saved_unique_labels)
        expected_decoding = ["rotate", "rotate", "rotate", "swipe_right"]
        encoding = np.array([
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                ])

        result = encoder.decode(encoding)

        np.testing.assert_array_equal(result, expected_decoding)


if __name__ == '__main__':
    unittest.main()
