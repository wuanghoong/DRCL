import numpy as np
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from collections import namedtuple
import torch.nn as nn
import torch
from cfg.config import MambaConfig
from mamba.head import MambaClassificationHead

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

class MambaTextClassification(MambaLMHeadModel):
    def __init__(
            self,
            config: MambaConfig,
            num_class: int = 11,
            initializer_cfg=None,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__(config, initializer_cfg, device, dtype)
        self.gradient_checkpointing = False
        self.classification_head = MambaClassificationHead(d_model=config.d_model, num_classes=9)  # cora 6, citeseet 11, pubmed 4, wikics 13, photo 10, arxiv_2023 35, history 6
        del self.lm_head

    def gradient_checkpointing_enable(self, **kwargs):
        """启用梯度检查点"""
        self.gradient_checkpointing = True
        if hasattr(self.backbone, 'gradient_checkpointing_enable'):
            # 传递所有kwargs给backbone
            self.backbone.gradient_checkpointing_enable(**kwargs)
    #
    # def forward(self, input_ids, attention_mask=None, labels=None):
    #     # Pass input_ids through the backbone model to receive hidden_states.
    #     hidden_states = self.backbone(input_ids)
    #
    #     # Take the mean of hidden_states along the second dimension to create a representative [CLS] feature.
    #     mean_hidden_states = hidden_states.mean(dim=1)
    #
    #     # Pass mean_hidden_states through the classification head to get logits.
    #     logits = self.classification_head(mean_hidden_states)
    #
    #     if labels is None:
    #         ClassificationOuptput = namedtuple("ClassificationOutput", ["logits", "text_feature"])
    #         return ClassificationOuptput(logits=logits, text_feature=mean_hidden_states)
    #     else:
    #         ClassificationOuptput = namedtuple("ClassificationOutput", ["loss", "logits", "text_feature"])
    #
    #         # Use CrossEntropyLoss loss function to compute the loss.
    #         loss_fct = nn.CrossEntropyLoss()
    #         loss = loss_fct(logits, labels)
    #
    #         return ClassificationOuptput(loss=loss, logits=logits, text_feature=mean_hidden_states)
    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(inputs[0])

                return custom_forward

            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.backbone),
                input_ids,
                use_reentrant=False
            )
        else:
            hidden_states = self.backbone(input_ids)

        mean_hidden_states = hidden_states.mean(dim=1)
        logits = self.classification_head(mean_hidden_states)

        if labels is None:
            ClassificationOutput = namedtuple("ClassificationOutput", ["logits"])
            return ClassificationOutput(logits=logits)
        else:
            ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return ClassificationOutput(loss=loss, logits=logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name, num_class: int = 6, device=None, dtype=None, **kwargs):
        # 加载配置
        config_data = load_config_hf(pretrained_model_name)
        print(config_data)

        # 转换参数命名（如有必要）
        if "hidden_size" in config_data:  # 兼容不同命名
            config_data["d_model"] = config_data.pop("hidden_size")
        if "num_hidden_layers" in config_data:
            config_data["n_layer"] = config_data.pop("num_hidden_layers")

        # 初始化配置
        config = MambaConfig(**config_data)

        # 加载模型
        model = cls(config, num_class=num_class, device=device, dtype=dtype, **kwargs)
        model_state_dict = load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype)
        model.load_state_dict(model_state_dict, strict=False)
        # model_state_dict = load_file(hf_hub_download("LeoYML/biomamba-130m", filename="model.safetensors"))
        # model.load_state_dict(model_state_dict)
        return model.to(device)