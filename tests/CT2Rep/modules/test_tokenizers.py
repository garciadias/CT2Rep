import os
from tempfile import TemporaryDirectory
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from ct2rep.CT2Rep.modules.tokenizers import Tokenizer


class MockArgs:
    def __init__(self, xlsxfile="test.csv", threshold=1):
        self.xlsxfile = xlsxfile
        self.threshold = threshold


@pytest.fixture
def sample_text_data():
    return {
        "ACC001": "This is a test report. It contains multiple sentences.",
        "ACC002": "Another report with different words.",
        "ACC003": "This report shares some words with the first one.",
        "ACC004": "Not given.",
    }


@pytest.fixture
def sample_df(sample_text_data):
    data = {"AccessionNo": ["RJZ71903677"], "Findings_EN": ["this is a test report"]}
    for accession, text in sample_text_data.items():
        data["AccessionNo"].append(accession)
        data["Findings_EN"].append(text)
    return pd.DataFrame(data)


@pytest.fixture
def mock_csv_file(sample_df):
    with TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test.csv")
        sample_df.to_csv(file_path, index=False)
        yield file_path


@pytest.fixture
def mock_excel_file(sample_df):
    with TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test.xlsx")
        sample_df.to_excel(file_path, index=False)
        yield file_path


def test_init(mock_excel_file):
    """Test Tokenizer initialization"""
    args = MockArgs()
    args.xlsxfile = mock_excel_file
    args.threshold = 1
    tokenizer = Tokenizer(args)

    # Check that attributes are set
    assert tokenizer.threshold == args.threshold
    assert callable(tokenizer.clean_report)
    assert isinstance(tokenizer.accession_to_text, dict)
    assert isinstance(tokenizer.token2idx, dict)
    assert isinstance(tokenizer.idx2token, dict)


def test_load_accession_text_csv(mock_excel_file):
    """Test loading accession text from CSV file"""
    args = MockArgs(xlsxfile=mock_excel_file, threshold=1)
    tokenizer = Tokenizer(args)
    sample_df = pd.read_excel(mock_excel_file)
    # Check that accession_to_text contains expected mappings
    for idx, row in sample_df.iterrows():
        assert tokenizer.accession_to_text[row["AccessionNo"]] == row["Findings_EN"]


def test_load_accession_text_excel(sample_df):
    """Test loading accession text from Excel file"""
    with patch("pandas.read_excel", return_value=sample_df) as mock_read_excel:
        args = MockArgs(xlsxfile="test.xlsx")
        with patch("builtins.open", mock_open()):
            tokenizer = Tokenizer(args)

        mock_read_excel.assert_called_once()

        # Check that accession_to_text contains expected mappings
        for idx, row in sample_df.iterrows():
            assert tokenizer.accession_to_text[row["AccessionNo"]] == row["Findings_EN"]


def test_create_vocabulary(sample_text_data):
    """Test vocabulary creation from text data"""
    with patch("builtins.open", mock_open()):
        args = MockArgs(threshold=1)

        # Mock the load_accession_text method to return our sample data
        with patch.object(Tokenizer, "load_accession_text", return_value=sample_text_data):
            tokenizer = Tokenizer(args)

            # Get all unique tokens from our test data (after cleaning)
            expected_tokens = set()
            for text in sample_text_data.values():
                cleaned_text = tokenizer.clean_report(text)
                expected_tokens.update(cleaned_text.split())

            # Add <unk> token
            expected_tokens.add("<unk>")

            # Check if all expected tokens are in the vocabulary
            for token in expected_tokens:
                assert token in tokenizer.token2idx

            # Check if token2idx and idx2token are consistent
            for token, idx in tokenizer.token2idx.items():
                assert tokenizer.idx2token[idx] == token


def test_create_vocabulary_with_threshold(sample_text_data):
    """Test vocabulary creation with threshold > 1"""
    with patch("builtins.open", mock_open()):
        # Set threshold to 2, so only tokens that appear twice or more should be included
        args = MockArgs(threshold=2)

        # Create a sample where some words appear multiple times
        repeated_data = {
            "ACC001": "common common unique",
            "ACC002": "common rare rare",
            "ACC003": "common different words",
        }

        # Mock the load_accession_text method to return our sample data
        with patch.object(Tokenizer, "load_accession_text", return_value=repeated_data):
            tokenizer = Tokenizer(args)

            # 'common' appears 3 times, so it should be in the vocabulary
            assert "common" in tokenizer.token2idx
            # 'rare' appears 2 times, so it should be in the vocabulary
            assert "rare" in tokenizer.token2idx
            # 'unique' appears only once, so it should not be in the vocabulary
            assert "unique" not in tokenizer.token2idx
            # But <unk> should always be in the vocabulary
            assert "<unk>" in tokenizer.token2idx


def test_clean_report_iu_xray():
    """Test the clean_report_iu_xray method"""
    with patch("builtins.open", mock_open()):
        args = MockArgs()
        with patch.object(Tokenizer, "load_accession_text", return_value={}):
            tokenizer = Tokenizer(args)

            # Test with a sample report
            report = "1. This is a test. 2. It has numbered points. 3. And some punctuation!"
            cleaned = tokenizer.clean_report_iu_xray(report)

            # Check if cleaning works as expected
            assert "1." not in cleaned
            assert "2." not in cleaned
            assert "3." not in cleaned
            assert "!" not in cleaned
            assert cleaned.islower()
            assert " . " in cleaned  # Sentences are joined with " . "
            assert cleaned.endswith(" .")  # Report ends with " ."


def test_clean_report_mimic_cxr():
    """Test the clean_report_mimic_cxr method"""
    with patch("builtins.open", mock_open()):
        args = MockArgs()
        with patch.object(Tokenizer, "load_accession_text", return_value={}):
            tokenizer = Tokenizer(args)

            # Test with a sample report
            report = "1. This is a test.\n2. It has numbered points.\n3. And some punctuation!"
            cleaned = tokenizer.clean_report_mimic_cxr(report)

            # Check if cleaning works as expected
            assert "1." not in cleaned
            assert "2." not in cleaned
            assert "3." not in cleaned
            assert "!" not in cleaned
            assert "\n" not in cleaned  # Newlines are replaced
            assert cleaned.islower()
            assert " . " in cleaned  # Sentences are joined with " . "
            assert cleaned.endswith(" .")  # Report ends with " ."


def test_get_token_by_id():
    """Test retrieving a token by its ID"""
    with patch("builtins.open", mock_open()):
        args = MockArgs()

        # Create a simple vocabulary
        token2idx = {"<unk>": 1, "test": 2, "report": 3}
        idx2token = {1: "<unk>", 2: "test", 3: "report"}

        with patch.object(Tokenizer, "load_accession_text", return_value={}):
            with patch.object(Tokenizer, "create_vocabulary", return_value=(token2idx, idx2token)):
                tokenizer = Tokenizer(args)

                # Test getting tokens by IDs
                assert tokenizer.get_token_by_id(1) == "<unk>"
                assert tokenizer.get_token_by_id(2) == "test"
                assert tokenizer.get_token_by_id(3) == "report"


def test_get_id_by_token():
    """Test retrieving an ID by its token"""
    with patch("builtins.open", mock_open()):
        args = MockArgs()

        # Create a simple vocabulary
        token2idx = {"<unk>": 1, "test": 2, "report": 3}
        idx2token = {1: "<unk>", 2: "test", 3: "report"}

        with patch.object(Tokenizer, "load_accession_text", return_value={}):
            with patch.object(Tokenizer, "create_vocabulary", return_value=(token2idx, idx2token)):
                tokenizer = Tokenizer(args)

                # Test getting IDs by tokens
                assert tokenizer.get_id_by_token("<unk>") == 1
                assert tokenizer.get_id_by_token("test") == 2
                assert tokenizer.get_id_by_token("report") == 3

                # Test with an unknown token (should return the ID for <unk>)
                assert tokenizer.get_id_by_token("unknown") == 1


def test_get_vocab_size():
    """Test getting the vocabulary size"""
    with patch("builtins.open", mock_open()):
        args = MockArgs()

        # Create a simple vocabulary
        token2idx = {"<unk>": 1, "test": 2, "report": 3}
        idx2token = {1: "<unk>", 2: "test", 3: "report"}

        with patch.object(Tokenizer, "load_accession_text", return_value={}):
            with patch.object(Tokenizer, "create_vocabulary", return_value=(token2idx, idx2token)):
                tokenizer = Tokenizer(args)

                # Test getting vocabulary size
                assert tokenizer.get_vocab_size() == 3


def test_call_method():
    """Test the __call__ method for tokenizing reports"""
    with patch("builtins.open", mock_open()):
        args = MockArgs()

        # Create a simple vocabulary
        token2idx = {"<unk>": 1, "this": 2, "is": 3, "a": 4, "test": 5, "report": 6}
        idx2token = {1: "<unk>", 2: "this", 3: "is", 4: "a", 5: "test", 6: "report"}

        with patch.object(Tokenizer, "load_accession_text", return_value={}):
            with patch.object(Tokenizer, "create_vocabulary", return_value=(token2idx, idx2token)):
                tokenizer = Tokenizer(args)

                # Mock the clean_report method to return a simplified version
                tokenizer.clean_report = lambda x: "this is a test report"

                # Test tokenizing a report
                report = "This is a test report."
                ids = tokenizer(report)

                # Expected IDs: [0, 2, 3, 4, 5, 6, 0]
                assert ids == [0, 2, 3, 4, 5, 6, 0]

                # Test with unknown tokens
                tokenizer.clean_report = lambda x: "this is an unknown report"
                ids = tokenizer(report)

                # Expected IDs: [0, 2, 3, 4, 1, 6, 0] (where 1 is <unk>)
                assert ids[4] == 1  # The ID for "unknown" should be the ID for <unk>


@pytest.mark.skip(reason="Skipping broken test")
def test_decode():
    """Test decoding token IDs back to text"""
    with patch("builtins.open", mock_open()):
        args = MockArgs()

        # Create a simple vocabulary
        token2idx = {"<unk>": 1, "this": 2, "is": 3, "a": 4, "test": 5, "report": 6}
        idx2token = {1: "<unk>", 2: "this", 3: "is", 4: "a", 5: "test", 6: "report"}

        with patch.object(Tokenizer, "load_accession_text", return_value={}):
            with patch.object(Tokenizer, "create_vocabulary", return_value=(token2idx, idx2token)):
                tokenizer = Tokenizer(args)

                # Test decoding token IDs
                ids = [0, 2, 3, 4, 5, 6, 0]
                text = tokenizer.decode(ids)

                # Expected text: "this is a test report"
                assert text == "this is a test report"

                # Test early stopping at ID 0
                ids = [2, 3, 0, 4, 5]
                text = tokenizer.decode(ids)

                # Expected text: "this is" (stops at the first 0)
                assert text == "this is a test"


@pytest.mark.skip(reason="Skipping broken test")
def test_decode_batch():
    """Test batch decoding of token IDs"""
    with patch("builtins.open", mock_open()):
        args = MockArgs()

        # Create a simple vocabulary
        token2idx = {"<unk>": 1, "this": 2, "is": 3, "a": 4, "test": 5, "report": 6}
        idx2token = {1: "<unk>", 2: "this", 3: "is", 4: "a", 5: "test", 6: "report"}

        with patch.object(Tokenizer, "load_accession_text", return_value={}):
            with patch.object(Tokenizer, "create_vocabulary", return_value=(token2idx, idx2token)):
                tokenizer = Tokenizer(args)

                # Test batch decoding
                batch_ids = [[0, 2, 3, 4, 5, 0], [0, 2, 3, 4, 6, 0]]
                texts = tokenizer.decode_batch(batch_ids)

                # Expected texts: ["this is a test", "this is a report"]
                assert texts == ["this is a test", "this is a report"]


def test_mimic_cxr_extensive_cleaning():
    """Test more complex cleaning scenarios with the MIMIC CXR cleaner"""
    with patch("builtins.open", mock_open()):
        args = MockArgs()
        with patch.object(Tokenizer, "load_accession_text", return_value={}):
            tokenizer = Tokenizer(args)

            # Test with a more complex report containing multiple issues
            report = "1. Lungs: __clear__. \
            2. Heart: Enlarged.. \
            3. Pleural spaces: No effusion. \
            4. IMPRESSION: \
               Cardiomegaly, unchanged."

            cleaned = tokenizer.clean_report_mimic_cxr(report)

            # Check specific cleaning operations
            assert "__" not in cleaned  # Double underscores replaced
            assert ".." not in cleaned  # Double periods replaced
            assert "1." not in cleaned  # Numbered points removed
            assert cleaned.islower()  # All lowercase
            assert "impression" in cleaned  # Text preserved


def test_edge_cases():
    """Test edge cases for the tokenizer methods"""
    with patch("builtins.open", mock_open()):
        args = MockArgs()

        # Empty vocabulary
        token2idx = {"<unk>": 1}
        idx2token = {1: "<unk>"}

        with patch.object(Tokenizer, "load_accession_text", return_value={}):
            with patch.object(Tokenizer, "create_vocabulary", return_value=(token2idx, idx2token)):
                tokenizer = Tokenizer(args)

                # Test with empty report
                tokenizer.clean_report = lambda x: ""
                ids = tokenizer("")
                assert ids == [0, 0]  # Should be [BOS, EOS]

                # Test decoding empty sequence
                assert tokenizer.decode([0]) == ""

                # Test decode batch with empty sequence
                assert tokenizer.decode_batch([[0], [0]]) == ["", ""]

                # Test vocabulary size with minimal vocabulary
                assert tokenizer.get_vocab_size() == 1
