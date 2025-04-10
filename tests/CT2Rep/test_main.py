import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ct2rep.CT2Rep.main import parse_agrs

MODULE_PATH = Path(__file__).resolve().parents[2]


class TestParseArgs:
    """Tests for the argument parser function."""

    def test_default_args(self):
        """Test that default arguments are set correctly."""
        # Mock sys.argv to simulate command line with no arguments
        with patch.object(sys, "argv", ["main.py"]):
            args = parse_agrs()

            # Check default values for various argument groups

            # Data loader settings
            assert args.max_seq_length == 200
            assert args.threshold == 3
            assert args.num_workers == 2
            assert args.batch_size == 2
            assert args.dataset_name == "ct_dataset"

            # Model settings
            assert args.d_model == 512
            assert args.d_ff == 512
            assert args.d_vf == 512
            assert args.num_heads == 8
            assert args.num_layers == 3
            assert args.dropout == 0.1
            assert args.logit_layers == 1
            assert args.bos_idx == 0
            assert args.eos_idx == 0
            assert args.pad_idx == 0
            assert args.use_bn == 0
            assert args.drop_prob_lm == 0.5

            # Relational Memory settings
            assert args.rm_num_slots == 3
            assert args.rm_num_heads == 8
            assert args.rm_d_model == 512

            # Sample related settings
            assert args.sample_method == "beam_search"
            assert args.beam_size == 3
            assert args.temperature == 1.0
            assert args.sample_n == 1
            assert args.group_size == 1
            assert args.output_logsoftmax == 1
            assert args.decoding_constraint == 0
            assert args.block_trigrams == 1

            # Trainer settings
            assert args.n_gpu == 1
            assert args.epochs == 100
            assert args.save_period == 1
            assert args.monitor_mode == "max"
            assert args.monitor_metric == "BLEU_4"
            assert args.early_stop == 50

            # Optimization settings
            assert args.optim == "Adam"
            assert args.lr_ve == 5e-5
            assert args.lr_ed == 1e-4
            assert args.weight_decay == 5e-5
            assert args.amsgrad is True

            # Learning Rate Scheduler settings
            assert args.lr_scheduler == "StepLR"
            assert args.step_size == 50
            assert args.gamma == 0.1

            # Path settings
            assert args.xlsxfile == f"{MODULE_PATH}/data/example_data/CT2Rep/data_reports_example.xlsx"
            assert args.trainfolder == f"{MODULE_PATH}/data/example_data/CT2Rep/train"
            assert args.validfolder == f"{MODULE_PATH}/data/example_data/CT2Rep/valid"

            # Resume training
            assert args.resume is None

    def test_custom_args(self):
        """Test parsing custom arguments."""
        test_args = [
            "main.py",
            "--max_seq_length",
            "300",
            "--threshold",
            "5",
            "--batch_size",
            "4",
            "--d_model",
            "768",
            "--num_heads",
            "12",
            "--epochs",
            "50",
            "--optim",
            "SGD",
            "--lr_ve",
            "0.001",
            "--resume",
            "checkpoint.pth",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_agrs()

            # Check custom values were parsed correctly
            assert args.max_seq_length == 300
            assert args.threshold == 5
            assert args.batch_size == 4
            assert args.d_model == 768
            assert args.num_heads == 12
            assert args.epochs == 50
            assert args.optim == "SGD"
            assert args.lr_ve == 0.001
            assert args.resume == "checkpoint.pth"

            # Check that unspecified args still have default values
            assert args.num_workers == 2
            assert args.dropout == 0.1

    def test_monitor_mode_choices(self):
        """Test that monitor_mode only accepts valid choices."""
        # Test valid choice 'min'
        with patch.object(sys, "argv", ["main.py", "--monitor_mode", "min"]):
            args = parse_agrs()
            assert args.monitor_mode == "min"

        # Test valid choice 'max'
        with patch.object(sys, "argv", ["main.py", "--monitor_mode", "max"]):
            args = parse_agrs()
            assert args.monitor_mode == "max"

        # Test invalid choice
        with patch.object(sys, "argv", ["main.py", "--monitor_mode", "invalid"]):
            with pytest.raises(SystemExit):
                parse_agrs()

    def test_path_arguments(self):
        """Test setting custom path arguments."""
        test_args = [
            "main.py",
            "--xlsxfile",
            "/custom/path/reports.xlsx",
            "--trainfolder",
            "/custom/data/train",
            "--validfolder",
            "/custom/data/valid",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_agrs()

            assert args.xlsxfile == "/custom/path/reports.xlsx"
            assert args.trainfolder == "/custom/data/train"
            assert args.validfolder == "/custom/data/valid"
