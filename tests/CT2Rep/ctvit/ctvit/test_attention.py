import pytest
import torch

# filepath: /ct2rep/src/ct2rep/CT2Rep/ctvit/ctvit/test_attention.py
import torch.nn.functional as F

from ct2rep.ctvit.ctvit.attention import (
    GEGLU,
    PEG,
    AlibiPositionalBias,
    Attention,
    ContinuousPositionBias,
    FeedForward,
    LayerNorm,
    Transformer,
    default,
    exists,
    l2norm,
)


@pytest.fixture
def device():
    """Return the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestHelperFunctions:
    """Test the helper utility functions in the attention module."""

    def test_exists(self):
        """Test the exists function with various inputs."""
        assert exists(1) is True
        assert exists(None) is False
        assert exists([]) is True
        assert exists("") is True

    def test_default(self):
        """Test the default function with various inputs."""
        assert default(None, 1) == 1
        assert default(2, 1) == 2
        assert default("", "default") == ""

    def test_l2norm(self, device):
        """Test the l2norm function for vector normalization."""
        # Test 1D vector normalization
        x = torch.tensor([3.0, 4.0], device=device)
        normalized = l2norm(x)
        assert torch.allclose(normalized, torch.tensor([0.6, 0.8], device=device))

        # Test 2D tensor normalization
        x = torch.tensor([[1.0, 2.0, 2.0], [3.0, 4.0, 0.0]], device=device)
        normalized = l2norm(x)

        norm1 = 3.0  # sqrt(1^2 + 2^2 + 2^2)
        norm2 = 5.0  # sqrt(3^2 + 4^2 + 0^2)
        expected1 = torch.tensor([1.0 / norm1, 2.0 / norm1, 2.0 / norm1], device=device)
        expected2 = torch.tensor([3.0 / norm2, 4.0 / norm2, 0.0 / norm2], device=device)

        assert torch.allclose(normalized, torch.stack([expected1, expected2]))


class TestLayerNorm:
    """Test the LayerNorm implementation."""

    def test_initialization(self):
        """Test LayerNorm initialization."""
        dim = 64
        norm = LayerNorm(dim)

        assert norm.gamma.shape == (dim,)
        assert norm.beta.shape == (dim,)
        assert torch.allclose(norm.gamma, torch.ones(dim))
        assert torch.allclose(norm.beta, torch.zeros(dim))

    # def test_forward(self, device):
    #     """Test LayerNorm forward pass."""
    #     dim = 64
    #     batch_size = 2
    #     seq_len = 10

    #     norm = LayerNorm(dim).to(device)
    #     x = torch.randn(batch_size, seq_len, dim, device=device)

    #     output = norm(x)

    #     # Check shape preservation
    #     assert output.shape == x.shape

    #     # Check normalization properties
    #     output_flat = output.reshape(-1, dim)
    #     mean = output_flat.mean(dim=0)
    #     var = output_flat.var(dim=0, unbiased=False)

    #     # Mean should be close to 0, variance close to 1
    #     assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    #     assert torch.allclose(var, torch.ones_like(var), atol=1e-5)


class TestGEGLU:
    """Test the GEGLU activation function."""

    def test_forward(self, device):
        """Test GEGLU forward pass."""
        batch_size = 2
        seq_len = 10
        dim = 64

        geglu = GEGLU().to(device)
        x = torch.randn(batch_size, seq_len, dim * 2, device=device)

        output = geglu(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, dim)

        # Manually compute expected output
        x_split, gate_split = x.chunk(2, dim=-1)
        expected = F.gelu(gate_split) * x_split

        assert torch.allclose(output, expected)


class TestFeedForward:
    """Test the FeedForward network."""

    def test_initialization(self):
        """Test FeedForward initialization with different parameters."""
        dim = 64

        # Default parameters
        ff = FeedForward(dim)
        assert len(ff) == 5  # LayerNorm + Linear + GEGLU + Dropout + Linear

        # With dropout
        ff_dropout = FeedForward(dim, dropout=0.1)
        assert ff_dropout[3].p == 0.1

        # Different multiplier
        ff_mult = FeedForward(dim, mult=2)
        assert ff_mult[1].out_features == int((2 * 2 / 3) * dim * 2)  # Inner dim calculation

    def test_forward(self, device):
        """Test FeedForward forward pass."""
        dim = 64
        batch_size = 2
        seq_len = 10

        ff = FeedForward(dim).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)

        output = ff(x)

        # Check shape preservation
        assert output.shape == x.shape


class TestAlibiPositionalBias:
    """Test the AlibiPositionalBias for positional encoding."""

    def test_initialization(self):
        """Test AlibiPositionalBias initialization."""
        heads = 8
        bias = AlibiPositionalBias(heads=heads)

        assert bias.heads == heads
        assert bias.slopes.shape == (heads, 1, 1)

    def test_get_slopes(self):
        """Test the slope calculation for different numbers of heads."""
        # Power of 2
        slopes_8 = AlibiPositionalBias._get_slopes(8)
        assert len(slopes_8) == 8
        # assert slopes_8[0] < slopes_8[1]  # Slopes should increase

        # Non-power of 2
        slopes_6 = AlibiPositionalBias._get_slopes(6)
        assert len(slopes_6) == 6

    def test_forward(self, device):
        """Test AlibiPositionalBias forward pass."""
        heads = 4
        seq_len = 10

        bias = AlibiPositionalBias(heads=heads).to(device)
        sim = torch.randn(1, heads, seq_len, seq_len, device=device)

        output = bias(sim)

        # Check output shape
        assert output.shape == (heads, seq_len, seq_len)

        # Test properties of alibi bias:
        # 1. Values should be non-positive
        assert (output <= 0).all()

        # 2. Closer positions should have higher (less negative) values
        for h in range(heads):
            for i in range(seq_len - 1):
                # Position i+1 should be more penalized than position i
                assert output[h, 0, i] > output[h, 0, i + 1]


class TestPEG:
    """Test the Position Embedding Generator (PEG)."""

    def test_initialization(self):
        """Test PEG initialization."""
        dim = 64

        # Default parameters
        peg = PEG(dim=dim)
        assert peg.causal is False
        assert peg.dsconv.in_channels == dim
        assert peg.dsconv.out_channels == dim
        assert peg.dsconv.groups == dim  # Depthwise convolution

        # With causal=True
        peg_causal = PEG(dim=dim, causal=True)
        assert peg_causal.causal is True

    def test_forward(self, device):
        """Test PEG forward pass with different input formats."""
        dim = 64
        batch_size = 2
        frames = 4
        height = 8
        width = 8

        peg = PEG(dim=dim).to(device)

        # Test with 5D input (batch, frames, height, width, dim)
        x_5d = torch.randn(batch_size, frames, height, width, dim, device=device)
        output_5d = peg(x_5d)

        # Check shape preservation
        assert output_5d.shape == x_5d.shape

        # Test with 3D input and shape parameter
        x_3d = torch.randn(batch_size, frames * height * width, dim, device=device)
        output_3d = peg(x_3d, shape=(batch_size, frames, height, width))

        # Check shape preservation
        assert output_3d.shape == x_3d.shape


class TestAttention:
    """Test the Attention mechanism."""

    @pytest.fixture
    def attention_params(self):
        """Return common parameters for creating attention instances."""
        return {"dim": 64, "dim_head": 32, "heads": 2}

    def test_initialization(self, attention_params):
        """Test Attention initialization with different parameters."""
        # Default parameters
        attn = Attention(**attention_params)
        assert attn.heads == 2
        assert attn.scale == 8
        assert attn.causal is False
        assert attn.num_null_kv == 0
        assert not hasattr(attn, "rel_pos_bias")

        # With causal=True
        attn_causal = Attention(**attention_params, causal=True)
        assert attn_causal.causal is True
        assert hasattr(attn_causal, "rel_pos_bias")

        # With null kv
        attn_null_kv = Attention(**attention_params, num_null_kv=2)
        assert attn_null_kv.num_null_kv == 2
        assert attn_null_kv.null_kv.shape == (2, 4, 32)  # heads, 2*num_null_kv, dim_head

        # With context normalization disabled
        attn_no_norm = Attention(**attention_params, norm_context=False)
        assert isinstance(attn_no_norm.context_norm, torch.nn.Identity)

    def test_forward_basic(self, attention_params, device):
        """Test basic forward pass with self-attention."""
        batch_size = 2
        seq_len = 10
        dim = attention_params["dim"]

        attn = Attention(**attention_params).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)

        output = attn(x)

        # Check shape preservation
        assert output.shape == x.shape

    def test_forward_with_context(self, attention_params, device):
        """Test forward pass with cross-attention."""
        batch_size = 2
        seq_len = 10
        context_len = 8
        dim = attention_params["dim"]
        context_dim = 48

        attn = Attention(**attention_params, dim_context=context_dim).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)
        context = torch.randn(batch_size, context_len, context_dim, device=device)

        output = attn(x, context=context)

        # Check shape preservation
        assert output.shape == x.shape

    def test_forward_with_mask(self, attention_params, device):
        """Test forward pass with attention mask."""
        batch_size = 2
        seq_len = 10
        dim = attention_params["dim"]

        attn = Attention(**attention_params).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)

        # Create a mask (False = masked positions)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        mask[:, :3] = False  # Mask first 3 positions

        output = attn(x, mask=mask)

        # Check shape preservation
        assert output.shape == x.shape

    def test_forward_with_attn_bias(self, attention_params, device):
        """Test forward pass with attention bias."""
        batch_size = 2
        seq_len = 10
        heads = attention_params["heads"]
        dim = attention_params["dim"]

        attn = Attention(**attention_params).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)

        # Create attention bias
        attn_bias = torch.randn(batch_size, heads, seq_len, seq_len, device=device)

        output = attn(x, attn_bias=attn_bias)

        # Check shape preservation
        assert output.shape == x.shape

    def test_forward_causal(self, attention_params, device):
        """Test forward pass with causal attention."""
        batch_size = 2
        seq_len = 10
        dim = attention_params["dim"]

        attn = Attention(**attention_params, causal=True).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)

        output = attn(x)

        # Check shape preservation
        assert output.shape == x.shape

        # More detailed test: early tokens shouldn't attend to later tokens
        # This would require examining the attention weights directly


class TestTransformer:
    """Test the Transformer architecture."""

    @pytest.fixture
    def transformer_params(self):
        """Return common parameters for creating transformer instances."""
        return {"dim": 64, "depth": 2, "heads": 4}

    def test_initialization(self, transformer_params):
        """Test Transformer initialization with different parameters."""
        # Default parameters
        transformer = Transformer(**transformer_params)
        assert len(transformer.layers) == transformer_params["depth"]

        # Check layer components
        for layer in transformer.layers:
            assert layer[0] is None  # No PEG by default
            assert isinstance(layer[1], Attention)  # Self-attention
            assert layer[2] is None  # No cross-attention by default
            assert isinstance(layer[3], torch.nn.Sequential)  # FeedForward

        # With PEG
        transformer_peg = Transformer(**transformer_params, peg=True)
        assert isinstance(transformer_peg.layers[0][0], PEG)

        # With cross-attention
        transformer_cross = Transformer(**transformer_params, has_cross_attn=True)
        assert isinstance(transformer_cross.layers[0][2], Attention)

    def test_forward_basic(self, transformer_params, device):
        """Test basic forward pass through transformer."""
        batch_size = 2
        seq_len = 10
        dim = transformer_params["dim"]

        transformer = Transformer(**transformer_params).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)

        output = transformer(x)

        # Check shape preservation
        assert output.shape == x.shape

    def test_forward_with_cross_attn(self, transformer_params, device):
        """Test forward pass with cross-attention."""
        batch_size = 2
        seq_len = 10
        context_len = 8
        dim = transformer_params["dim"]
        context_dim = 32

        transformer = Transformer(**transformer_params, has_cross_attn=True, dim_context=context_dim).to(device)

        x = torch.randn(batch_size, seq_len, dim, device=device)
        context = torch.randn(batch_size, context_len, context_dim, device=device)

        output = transformer(x, context=context)

        # Check shape preservation
        assert output.shape == x.shape

    def test_forward_with_peg(self, transformer_params, device):
        """Test forward pass with PEG."""
        batch_size = 2
        frames = 4
        height = 8
        width = 8
        dim = transformer_params["dim"]

        transformer = Transformer(**transformer_params, peg=True).to(device)

        x = torch.randn(batch_size, frames * height * width, dim, device=device)
        video_shape = (batch_size, frames, height, width)

        output = transformer(x, video_shape=video_shape)

        # Check shape preservation
        assert output.shape == x.shape

    def test_forward_with_masks(self, transformer_params, device):
        """Test forward pass with attention masks."""
        batch_size = 2
        seq_len = 10
        dim = transformer_params["dim"]

        transformer = Transformer(**transformer_params).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)

        # Create self-attention mask
        self_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        self_mask[:, :3] = False  # Mask first 3 positions

        output = transformer(x, self_attn_mask=self_mask)

        # Check shape preservation
        assert output.shape == x.shape

        # Test with cross-attention
        transformer_cross = Transformer(**transformer_params, has_cross_attn=True).to(device)
        context = torch.randn(batch_size, seq_len, dim, device=device)

        # Create cross-attention mask
        cross_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        cross_mask[:, 5:] = False  # Mask last 5 positions

        output = transformer_cross(x, context=context, self_attn_mask=self_mask, cross_attn_context_mask=cross_mask)

        # Check shape preservation
        assert output.shape == x.shape


class TestContinuousPositionBias:
    """Test the ContinuousPositionBias for relative position encoding."""

    def test_initialization(self):
        """Test ContinuousPositionBias initialization."""
        dim = 32
        heads = 4

        # Default parameters
        cpb = ContinuousPositionBias(dim=dim, heads=heads)
        assert cpb.num_dims == 2
        assert cpb.log_dist is True
        assert len(cpb.net) == 3  # input layer, hidden layer, output layer

        # Custom parameters
        cpb_custom = ContinuousPositionBias(dim=dim, heads=heads, num_dims=3, layers=3, log_dist=False)
        assert cpb_custom.num_dims == 3
        assert cpb_custom.log_dist is False
        assert len(cpb_custom.net) == 4  # input layer, 2 hidden layers, output layer

    def test_forward(self, device):
        """Test ContinuousPositionBias forward pass."""
        dim = 32
        heads = 4
        height = 6
        width = 6

        cpb = ContinuousPositionBias(dim=dim, heads=heads).to(device)

        # Test with 2D dimensions
        output = cpb(height, width, device=device)

        # Check output shape
        assert output.shape == (heads, height * width, height * width)

        # Test with caching enabled
        cpb.cache_rel_pos = True
        output_cached = cpb(height, width, device=device)

        # Output should be identical after caching
        assert torch.allclose(output, output_cached)
        assert exists(cpb.rel_pos)
