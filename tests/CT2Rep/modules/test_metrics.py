from unittest.mock import Mock, patch

import pytest

from ct2rep.CT2Rep.modules.metrics import compute_scores

# Import the function to test using relative importmport pytest


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    gts = {
        'img1': [
            'A person cycling through the park on a sunny day.',
            'A person cycling through the park on a sunny day.'
        ],
        'img2': [
            'A dog running in the park, chasing a ball.',
            'A cat running in the park, chasing a ball.'
        ],
        'img3': [
            'A bus parked on the side of the street near houses.',
            'A van parked on the side of the street near houses.'
        ]
    }

    res = {
        'img1': ['A person cycling through the park on a sunny day.'],
        'img2': ['A dog running in the park, chasing a ball.'],
        'img3': ['A car parked on the side of the street near houses.']
    }

    return gts, res


@pytest.fixture
def mock_bleu_scores():
    """Sample BLEU scores for mocking"""
    return [0.85, 0.75, 0.65, 0.55]


@pytest.mark.parametrize(
    ('gts', 'res', 'expected'),
    [
        (
            {'img1': ['A person riding a bike on a sunny day.']},
            {'img1': ['A person riding a bike on a sunny day.']},
            [1, 1, 1, 1]
        ),
        (
            {'img2': ['A dog running in the park.']},
            {'img2': ['A dog chasing a ball in the park.']},
            [0.625, 0.517, 0.355, 0.01]
        ),
        (
            {'img3': ['A car parked on the street.']},
            {'img3': ['A van parked on the street.']},
            [0.833, 0.707, 0.630, 0.500]
        )
    ]
)
def test_bleu_scorer_for_each_image(gts, res, expected):
    """Test if BLEU scores are computed correctly for each image"""

    result = compute_scores(gts, res)

    # Check if BLEU scores were correctly processed
    assert result['BLEU_1'] == pytest.approx(expected[0], rel=1e-2)
    assert result['BLEU_2'] == pytest.approx(expected[1], rel=1e-2)
    assert result['BLEU_3'] == pytest.approx(expected[2], rel=1e-2)
    assert result['BLEU_4'] == pytest.approx(expected[3], rel=1)


def test_compute_scores_basic(sample_data):
    """Test basic functionality with mocked Bleu scorer"""
    gts, res = sample_data

    result = compute_scores(gts, res)

    # Check if BLEU scores were correctly processed
    assert round(result['BLEU_1'], 3) == pytest.approx(0.967)
    assert round(result['BLEU_2'], 3) == pytest.approx(0.946)
    assert round(result['BLEU_3'], 3) == pytest.approx(0.936)
    assert round(result['BLEU_4'], 3) == pytest.approx(0.928)
    assert len(result) == 4  # Only 4 BLEU scores


def test_compute_scores_empty():
    """Test with empty input dictionaries"""
    gts, res = {}, {}

    # Create mock for Bleu
    mock_bleu = Mock()
    mock_bleu.compute_score.return_value = ([0, 0, 0, 0], None)

    with patch('pycocoevalcap.bleu.bleu.Bleu', return_value=mock_bleu):
        result = compute_scores(gts, res)

    # Check if function handles empty inputs
    assert all(result[key] == 0 for key in result)
    assert len(result) == 4  # Still should return 4 BLEU scores


def test_compute_scores_real(sample_data):
    """Integration test with actual Bleu computation"""
    gts, res = sample_data

    # This test uses the actual Bleu implementation
    result = compute_scores(gts, res)

    # Verify we got expected result structure
    assert "BLEU_1" in result
    assert "BLEU_2" in result
    assert "BLEU_3" in result
    assert "BLEU_4" in result
    assert all(0 <= result[k] <= 1 for k in result)  # BLEU scores are between 0 and 1


def test_compute_scores_different_data():
    """Test with different data distributions"""
    # Test with longer sentences
    gts = {
        'img1': ['This is a much longer sentence with more words to test the robustness of the scoring function.'],
        'img2': ['Another detailed description that contains multiple clauses and technical terminology.']
    }

    res = {
        'img1': ['A longer sentence with many words for testing.'],
        'img2': ['Description with technical terminology.']
    }

    result = compute_scores(gts, res)

    # Just verify we get valid scores
    assert all(0 <= result[k] <= 1 for k in result)

    # Test with very short sentences
    gts = {'img1': ['Short.'], 'img2': ['Brief text.']}
    res = {'img1': ['Short text.'], 'img2': ['Brief.']}

    result = compute_scores(gts, res)
    assert all(0 <= result[k] <= 1 for k in result)


if __name__ == "__main__":
    pytest.main([__file__])
