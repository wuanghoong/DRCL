import json
from dataclasses import dataclass , field , asdict


@dataclass
class MambaConfig:
    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple:int = 8
    tie_embeddings: bool = True

    # 新增参数（来自biomamba-130m）
    d_inner: int = 1536
    state_size: int = 16
    time_step_rank: int = 48
    conv_kernel: int = 4
    expand: int = 2
    hidden_act: str = "silu"
    layer_norm_epsilon: float = 1e-5
    pad_token_id: int = 0
    bos_token_id: int = 0
    eos_token_id: int = 0
    torch_dtype: str = "float32"
    initializer_range: int = 0.1
    intermediate_size: int = 1536
    rescale_prenorm_residual: str = False
    time_step_floor: int = 0.0001
    time_step_init_scheme: str = 'random'
    time_step_max: int = 0.1
    time_step_min: int = 0.001
    time_step_scale: int = 1.0
    use_bias: str = False
    use_cache: str = True
    use_conv_bias: str = True

    # Hugging Face专用字段
    _name_or_path: str = None
    architectures: list = None
    model_type: str = "mamba"
    transformers_version: str = None


    def to_json_string(self):
        return json.dumps(asdict(self))

    def to_dict(self):
        return asdict(self)
