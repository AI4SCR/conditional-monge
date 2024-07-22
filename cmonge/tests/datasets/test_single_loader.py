from typing import Iterator

import jax.numpy as jnp
import pytest

from cmonge.datasets.single_loader import get_train_valid_test_split


class TestTrainValidTestSplits:
    @pytest.fixture
    def array(self) -> jnp.ndarray:
        return jnp.array([[1], [2], [3], [4], [5]])

    def test_get_train_valid_test_split_normal(self, array):
        split = [0.6, 0.2, 0.2]

        train, valid, test = get_train_valid_test_split(array, split)

        assert train.shape == (3, 1)
        assert valid.shape == (1, 1)
        assert test.shape == (1, 1)

    def test_get_train_valid_test_split_no_train(self, array):
        split = [0, 0.8, 0.2]

        train, valid, test = get_train_valid_test_split(array, split)

        assert train.shape == (0,)
        assert valid.shape == (4, 1)
        assert test.shape == (1, 1)

    def test_get_train_valid_test_split_no_valid(self, array):
        split = [0, 0, 1]

        train, valid, test = get_train_valid_test_split(array, split)

        assert train.shape == (0,)
        assert valid.shape == (0,)
        assert test.shape == (5, 1)

    def test_get_train_valid_test_split_no_test(self, array):
        split = [0, 1, 0]

        train, valid, test = get_train_valid_test_split(array, split)

        assert train.shape == (0,)
        assert valid.shape == (5, 1)
        assert test.shape == (0,)


class TestSyntheticDosageModule:
    def test_train_loaders(self, synthetic_data):
        loaders = synthetic_data.train_dataloaders()
        assert isinstance(loaders[0], Iterator)
        assert isinstance(next(loaders[0]), jnp.ndarray)
