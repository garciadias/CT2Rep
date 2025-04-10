# from unittest.mock import MagicMock, patch

# import pytest
# import torch

# from ct2rep.CT2Rep.modules.att_model import AttModel


# @pytest.fixture
# def mock_tokenizer():
#     """Create a mock tokenizer with a simple vocabulary."""
#     tokenizer = MagicMock()
#     tokenizer.idx2token = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "word1", 4: "word2", 5: "word3"}
#     return tokenizer


# @pytest.fixture
# def model_args():
#     """Create arguments for the AttModel."""
#     args = MagicMock()
#     args.d_model = 512
#     args.d_ff = 512
#     args.d_vf = 512
#     args.num_layers = 3
#     args.drop_prob_lm = 0.1
#     args.max_seq_length = 20
#     args.bos_idx = 1
#     args.eos_idx = 2
#     args.pad_idx = 0
#     args.use_bn = 0
#     return args


# @pytest.fixture
# def mock_prepare_feature():
#     """Mock the _prepare_feature method to return predictable values."""

#     def mock_prep(fc_feats, att_feats, att_masks):
#         batch_size = fc_feats.size(0)
#         dim = 512  # Using fixed dimension for testing
#         p_fc_feats = fc_feats  # Pass through
#         p_att_feats = torch.ones((batch_size, 3, dim))  # Simplified att_feats
#         pp_att_feats = torch.ones((batch_size, 3, dim))  # Same shape as p_att_feats
#         p_att_masks = torch.ones((batch_size, 3)) if att_masks is not None else None
#         return p_fc_feats, p_att_feats, pp_att_feats, p_att_masks

#     return mock_prep


# @pytest.fixture
# def mock_beam_search():
#     """Mock the beam_search method to return predictable beams."""

#     def mock_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt={}):
#         batch_size = state.size(0)
#         beam_size = opt.get("beam_size", 3)
#         vocab_size = logprobs.size(1)

#         # Create a list of beams for each batch item
#         done_beams = []
#         for i in range(batch_size):
#             batch_beams = []
#             for j in range(beam_size):
#                 # Create beams with predictable sequences and logprobs
#                 seq_len = 5  # Use a fixed sequence length
#                 # Create a sequence: [bos_idx, j+1, j+2, j+3, eos_idx]
#                 seq = torch.tensor([1] + list(range(j + 1, j + seq_len - 1)) + [2])
#                 # Create logprobs for each token (just use a constant value)
#                 logps = torch.ones(seq.size(0)) * (-0.5 * (j + 1))  # Higher beams have lower logprobs
#                 batch_beams.append({"seq": seq, "logps": logps})
#             done_beams.append(batch_beams)
#         return done_beams

#     return mock_search


# @pytest.fixture
# def setup_model(mock_tokenizer, model_args, mock_prepare_feature, mock_beam_search):
#     """Setup a testable AttModel."""

#     class TestableAttModel(AttModel):
#         def init_hidden(self, batch_size):
#             return torch.zeros(batch_size, self.rnn_size)

#         def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
#             # Return predictable logprobs
#             batch_size = it.size(0)
#             logprobs = torch.ones(batch_size, self.vocab_size + 1) * -1  # All tokens equally likely
#             # Make a few tokens more likely for testing
#             for i in range(batch_size):
#                 logprobs[i, i % 5 + 1] = 0  # Token i%5+1 is most likely
#             return logprobs, state

#         def core(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
#             return torch.zeros(xt.size(0), self.rnn_size), state

#         def ctx2att(self, ctx):
#             return ctx

#         def logit(self, output):
#             return output

#         def sample_next_word(self, logprobs, *args, **kwargs):
#             # Return the most likely word and its probability
#             it = logprobs.argmax(dim=1)
#             sampleLogprobs = torch.zeros_like(it, dtype=torch.float)
#             return it, sampleLogprobs

#     # Create the model
#     model = TestableAttModel(model_args, mock_tokenizer)

#     # Patch methods using monkeypatch directly on the instance
#     model._prepare_feature = lambda *args, **kwargs: mock_prepare_feature(*args[1:], **kwargs)
#     model.beam_search = lambda *args, **kwargs: mock_beam_search(*args, **kwargs)

#     return model


# def test_sample_beam_basic(setup_model):
#     """Test _sample_beam with default parameters."""
#     model = setup_model

#     # Test data
#     batch_size = 2
#     feat_dim = model.att_feat_size
#     fc_feats = torch.randn(batch_size, model.input_encoding_size)
#     att_feats = torch.randn(batch_size, 5, feat_dim)
#     att_masks = torch.ones(batch_size, 5)

#     # Default options
#     opt = {"beam_size": 3, "sample_n": 1}

#     # Call the method
#     seq, seqLogprobs = model._sample_beam(fc_feats, att_feats, att_masks, opt)

#     # Check shapes
#     assert seq.shape == (batch_size, model.max_seq_length)
#     assert seqLogprobs.shape == (batch_size, model.max_seq_length, model.vocab_size + 1)

#     # The first beam should be used for each batch item
#     for i in range(batch_size):
#         # First token should be BOS
#         assert seq[i, 0] == model.bos_idx
#         # Next tokens should follow the pattern from our mock beam_search
#         assert seq[i, 1] == 1  # First beam uses tokens starting from 1


# def test_sample_beam_with_sample_n_equal_beam_size(setup_model):
#     """Test _sample_beam with sample_n equal to beam_size."""
#     model = setup_model

#     # Test data
#     batch_size = 2
#     feat_dim = model.att_feat_size
#     fc_feats = torch.randn(batch_size, model.input_encoding_size)
#     att_feats = torch.randn(batch_size, 5, feat_dim)
#     att_masks = torch.ones(batch_size, 5)

#     # Set sample_n = beam_size
#     beam_size = 3
#     opt = {"beam_size": beam_size, "sample_n": beam_size}

#     # Call the method
#     seq, seqLogprobs = model._sample_beam(fc_feats, att_feats, att_masks, opt)

#     # Check shapes - should have batch_size * sample_n sequences
#     assert seq.shape == (batch_size * beam_size, model.max_seq_length)
#     assert seqLogprobs.shape == (batch_size * beam_size, model.max_seq_length, model.vocab_size + 1)

#     # Check that all beams are used
#     for i in range(batch_size):
#         for j in range(beam_size):
#             idx = i * beam_size + j
#             # First token should be BOS
#             assert seq[idx, 0] == model.bos_idx
#             # Second token should match the beam index (our mock returns j+1)
#             assert seq[idx, 1] == j + 1


# def test_sample_beam_assertion_error(setup_model):
#     """Test that _sample_beam raises an assertion error with invalid parameters."""
#     model = setup_model

#     # Test data
#     batch_size = 2
#     feat_dim = model.att_feat_size
#     fc_feats = torch.randn(batch_size, model.input_encoding_size)
#     att_feats = torch.randn(batch_size, 5, feat_dim)
#     att_masks = torch.ones(batch_size, 5)

#     # Set invalid sample_n (not 1 and not beam_size)
#     beam_size = 4
#     sample_n = 2
#     opt = {"beam_size": beam_size, "sample_n": sample_n}

#     # Should raise an assertion error
#     with pytest.raises(AssertionError):
#         model._sample_beam(fc_feats, att_feats, att_masks, opt)


# def test_sample_beam_large_beam_size(setup_model):
#     """Test that _sample_beam raises an assertion error with beam_size > vocab_size+1."""
#     model = setup_model

#     # Test data
#     batch_size = 2
#     feat_dim = model.att_feat_size
#     fc_feats = torch.randn(batch_size, model.input_encoding_size)
#     att_feats = torch.randn(batch_size, 5, feat_dim)
#     att_masks = torch.ones(batch_size, 5)

#     # Set beam_size larger than vocab_size+1
#     beam_size = model.vocab_size + 2
#     opt = {"beam_size": beam_size, "sample_n": 1}

#     # Should raise an assertion error
#     with pytest.raises(AssertionError):
#         model._sample_beam(fc_feats, att_feats, att_masks, opt)


# @patch("ct2rep.modules.utils.repeat_tensors")
# def test_repeat_tensors_called_correctly(mock_repeat_tensors, setup_model):
#     """Test that repeat_tensors is called with correct parameters."""
#     model = setup_model

#     # Configure the mock
#     def side_effect(beam_size, tensors):
#         # Just pass through for testing
#         return tensors

#     mock_repeat_tensors.side_effect = side_effect

#     # Test data
#     batch_size = 2
#     feat_dim = model.att_feat_size
#     fc_feats = torch.randn(batch_size, model.input_encoding_size)
#     att_feats = torch.randn(batch_size, 5, feat_dim)
#     att_masks = torch.ones(batch_size, 5)

#     # Call with specific beam_size
#     beam_size = 3
#     opt = {"beam_size": beam_size, "sample_n": 1}

#     model._sample_beam(fc_feats, att_feats, att_masks, opt)

#     # Verify repeat_tensors was called with correct beam_size
#     mock_repeat_tensors.assert_called_once()
#     args, _ = mock_repeat_tensors.call_args
#     assert args[0] == beam_size


# def test_sample_beam_without_masks(setup_model):
#     """Test _sample_beam without attention masks."""
#     model = setup_model

#     # Test data
#     batch_size = 2
#     feat_dim = model.att_feat_size
#     fc_feats = torch.randn(batch_size, model.input_encoding_size)
#     att_feats = torch.randn(batch_size, 5, feat_dim)

#     # Call without att_masks
#     opt = {"beam_size": 3, "sample_n": 1}

#     # Should not raise errors
#     seq, seqLogprobs = model._sample_beam(fc_feats, att_feats, None, opt)

#     # Check shapes
#     assert seq.shape == (batch_size, model.max_seq_length)
#     assert seqLogprobs.shape == (batch_size, model.max_seq_length, model.vocab_size + 1)
