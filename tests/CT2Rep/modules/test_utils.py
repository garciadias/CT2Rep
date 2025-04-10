import pytest
import torch

from ct2rep.CT2Rep.modules.utils import length_average, length_wu, penalty_builder, repeat_tensors, split_tensors


# Tests for penalty_builder
def test_penalty_builder_empty():
    """Test penalty_builder with empty string returns identity function."""
    penalty_fn = penalty_builder("")
    x, y = 5, 10
    assert penalty_fn(x, y) == y


def test_penalty_builder_wu():
    """Test penalty_builder creates Wu penalty function correctly."""
    penalty_fn = penalty_builder("wu_0.5")
    length, logprobs = 10, -20.0
    expected = length_wu(length, logprobs, 0.5)
    assert penalty_fn(length, logprobs) == expected


def test_penalty_builder_avg():
    """Test penalty_builder creates average penalty function correctly."""
    penalty_fn = penalty_builder("avg_1.0")
    length, logprobs = 5, -10.0
    expected = length_average(length, logprobs, 1.0)
    assert penalty_fn(length, logprobs) == expected


# Tests for length_wu
def test_length_wu_zero_alpha():
    """Test length_wu with alpha=0 returns the original logprobs."""
    length, logprobs = 10, -20.0
    result = length_wu(length, logprobs, 0.0)
    assert result == logprobs


def test_length_wu_with_alpha():
    """Test length_wu with non-zero alpha."""
    length, logprobs = 10, -20.0
    alpha = 0.5
    modifier = ((5 + length) ** alpha) / ((5 + 1) ** alpha)
    expected = logprobs / modifier
    result = length_wu(length, logprobs, alpha)
    assert result == pytest.approx(expected)


# Tests for length_average
def test_length_average_basic():
    """Test length_average divides logprobs by length."""
    length, logprobs = 5, -10.0
    result = length_average(length, logprobs, 1.0)
    assert result == logprobs / length


def test_length_average_zero_alpha():
    """Test length_average with alpha=0 still divides by length."""
    length, logprobs = 5, -10.0
    result = length_average(length, logprobs, 0.0)
    assert result == logprobs / length


# Tests for split_tensors
def test_split_tensors_tensor():
    """Test split_tensors with a simple tensor."""
    x = torch.arange(12).reshape(6, 2)
    result = split_tensors(3, x)
    assert len(result) == 3
    assert torch.all(result[0] == torch.tensor([[0, 1], [6, 7]]))
    assert torch.all(result[1] == torch.tensor([[2, 3], [8, 9]]))
    assert torch.all(result[2] == torch.tensor([[4, 5], [10, 11]]))


def test_split_tensors_list():
    """Test split_tensors with a list of tensors."""
    x = [torch.arange(6).reshape(3, 2), torch.arange(12).reshape(6, 2)]
    result = split_tensors(3, x)
    assert len(result) == 2  # Still a list with 2 elements
    assert len(result[0]) == 3  # Each element now has 3 tensors
    assert len(result[1]) == 3

    # Check first list element's splits
    assert torch.all(result[0][0] == torch.tensor([[0, 1]]))
    assert torch.all(result[0][1] == torch.tensor([[2, 3]]))
    assert torch.all(result[0][2] == torch.tensor([[4, 5]]))

    # Check second list element's splits
    assert torch.all(result[1][0] == torch.tensor([[0, 1], [6, 7]]))
    assert torch.all(result[1][1] == torch.tensor([[2, 3], [8, 9]]))
    assert torch.all(result[1][2] == torch.tensor([[4, 5], [10, 11]]))


def test_split_tensors_tuple():
    """Test split_tensors with a tuple of tensors."""
    x = (torch.arange(6).reshape(3, 2), torch.arange(12).reshape(6, 2))
    result = split_tensors(3, x)
    assert len(result) == 2  # Still a tuple with 2 elements
    assert len(result[0]) == 3  # Each element now has 3 tensors
    assert len(result[1]) == 3

    # Check first tuple element's splits
    assert torch.all(result[0][0] == torch.tensor([[0, 1]]))
    assert torch.all(result[0][1] == torch.tensor([[2, 3]]))
    assert torch.all(result[0][2] == torch.tensor([[4, 5]]))

    # Check second tuple element's splits
    assert torch.all(result[1][0] == torch.tensor([[0, 1], [6, 7]]))
    assert torch.all(result[1][1] == torch.tensor([[2, 3], [8, 9]]))
    assert torch.all(result[1][2] == torch.tensor([[4, 5], [10, 11]]))


def test_split_tensors_none():
    """Test split_tensors with None."""
    x = None
    result = split_tensors(3, x)
    assert result == [None, None, None]


def test_split_tensors_mixed_list():
    """Test split_tensors with a list containing tensors and None."""
    x = [torch.arange(6).reshape(3, 2), None]
    result = split_tensors(3, x)
    assert len(result) == 2
    assert len(result[0]) == 3
    assert result[1] == [None, None, None]

    # Check tensor split
    assert torch.all(result[0][0] == torch.tensor([[0, 1]]))
    assert torch.all(result[0][1] == torch.tensor([[2, 3]]))
    assert torch.all(result[0][2] == torch.tensor([[4, 5]]))


def test_split_tensors_error_on_invalid_batch():
    """Test split_tensors raises error when batch size is not divisible by n."""
    x = torch.arange(5)  # Batch size of 5
    with pytest.raises(AssertionError):
        split_tensors(3, x)  # 5 is not divisible by 3


# Tests for repeat_tensors
def test_repeat_tensors_tensor():
    """Test repeat_tensors with a simple tensor."""
    x = torch.arange(6).reshape(3, 2)
    result = repeat_tensors(2, x)
    assert result.shape == (6, 2)

    # Expected: each row is repeated consecutively
    expected = torch.tensor(
        [
            [0, 1],  # First row repeated
            [0, 1],
            [2, 3],  # Second row repeated
            [2, 3],
            [4, 5],  # Third row repeated
            [4, 5],
        ]
    )
    assert torch.all(result == expected)


def test_repeat_tensors_3d_tensor():
    """Test repeat_tensors with a 3D tensor."""
    x = torch.arange(12).reshape(2, 2, 3)
    result = repeat_tensors(3, x)
    assert result.shape == (6, 2, 3)

    # Check first batch item repeated 3 times
    for i in range(3):
        assert torch.all(result[i] == x[0])

    # Check second batch item repeated 3 times
    for i in range(3, 6):
        assert torch.all(result[i] == x[1])


def test_repeat_tensors_list():
    """Test repeat_tensors with a list of tensors."""
    x = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([5, 6])]
    result = repeat_tensors(2, x)

    # Check first list element
    assert result[0].shape == (4, 2)
    assert torch.all(result[0][:2] == torch.tensor([[1, 2], [1, 2]]))
    assert torch.all(result[0][2:] == torch.tensor([[3, 4], [3, 4]]))

    # Check second list element
    assert result[1].shape == (4,)
    assert torch.all(result[1][:2] == torch.tensor([5, 5]))


def test_repeat_tensors_tuple():
    """Test repeat_tensors with a tuple of tensors."""
    x = (torch.tensor([[1, 2]]), torch.tensor([3, 4, 5]))
    result = repeat_tensors(3, x)

    # Check first tuple element
    assert result[0].shape == (3, 2)
    assert torch.all(result[0] == torch.tensor([[1, 2], [1, 2], [1, 2]]))

    # Check second tuple element
    assert result[1].shape == (9,)
    assert torch.all(result[1] == torch.tensor([3, 3, 3, 4, 4, 4, 5, 5, 5]))


def test_repeat_tensors_empty_tensor():
    """Test repeat_tensors with an empty tensor."""
    x = torch.tensor([])
    result = repeat_tensors(3, x)
    assert result.shape[0] == 0  # Should remain empty


def test_repeat_tensors_nested_list():
    """Test repeat_tensors with nested lists containing tensors."""
    x = [torch.tensor([[1]]), [torch.tensor([2]), torch.tensor([3])]]
    result = repeat_tensors(2, x)

    # Check first outer list element
    assert result[0].shape == (2, 1)
    assert torch.all(result[0] == torch.tensor([[1], [1]]))

    # Check second outer list element (which is itself a list)
    assert len(result[1]) == 2
    assert torch.all(result[1][0] == torch.tensor([2, 2]))
    assert torch.all(result[1][1] == torch.tensor([3, 3]))


def test_repeat_tensors_scalar_tensor():
    """Test repeat_tensors with a scalar tensor."""
    x = torch.tensor(5)
    x = x.reshape(1)  # Make it a batch of 1
    result = repeat_tensors(4, x)
    assert result.shape == (4,)
    assert torch.all(result == torch.tensor([5, 5, 5, 5]))


# Integration tests
def test_repeat_and_split_are_inverses():
    """Test that split_tensors and repeat_tensors are inverse operations."""
    original = torch.arange(10).reshape(5, 2)
    n = 2

    # Apply repeat then split
    repeated = repeat_tensors(n, original)
    split_result = split_tensors(n, repeated)

    # Check that each split matches the original
    for i in range(n):
        assert torch.all(split_result[i] == original)
