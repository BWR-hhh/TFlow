from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel

from .config import ParameterGeneratorConfig
from .model import TransformerModel
from .tokenizer import Tokenizer2DBatchedLoRA


class ParameterGenerator(PreTrainedModel):
    config_class = ParameterGeneratorConfig
    _no_split_modules = ["TransformerBlock"]

    def __init__(self, config: ParameterGeneratorConfig):
        super().__init__(config)

        dm, hd, r, da = config.d_model, config.head_dim, config.rank, config.dim_accumulation
        if dm % hd != 0:
            raise ValueError(f"ParameterGenerator: d_model ({dm}) must be divisible by head_dim ({hd}).")
        if dm % da != 0:
            raise ValueError(f"ParameterGenerator: d_model ({dm}) must be divisible by dim_accumulation ({da}).")
        if r % da != 0:
            raise ValueError(f"ParameterGenerator: rank ({r}) must be divisible by dim_accumulation ({da}).")
        if r // da < 1:
            raise ValueError(
                f"ParameterGenerator: rank ({r}) // dim_accumulation ({da}) must be >= 1; "
                "e.g. rank=2 requires dim_accumulation in {{1, 2}}, not 4."
            )

        self.tokenizer = Tokenizer2DBatchedLoRA(
            token_dim=config.token_dim,
            rank=config.rank,
            alpha=config.alpha,
            pg_mapping=config.pg_mapping,
        )
        self.lora_A_token_count = self.tokenizer.lora_A_token_count
        self.lora_B_token_count = self.tokenizer.lora_B_token_count

        self.model = TransformerModel(
            d_model=config.d_model,
            num_base_model_layers=config.num_base_model_layers,
            num_token_per_layer=self.lora_A_token_count + self.lora_B_token_count,
            lora_rank=config.rank,
            output_dim=config.output_dim,
            head_dim=config.head_dim,
            num_blocks=config.num_pg_layers,
            dim_accumulation=config.dim_accumulation,
            lora_A_token_count=self.lora_A_token_count,
            lora_B_token_count=self.lora_B_token_count,
        )

        self.layer_num = config.num_base_model_layers
        self.prefix = config.prefix
        self.lora_rank = config.rank

        self.hidden_in = nn.Linear(config.input_dim, config.d_model)

    def get_lora_count(self):
        return (
            self.config.output_dim
            * (self.lora_A_token_count + self.lora_B_token_count)
            * self.config.num_base_model_layers
            * self.lora_rank
        )

    def forward(self, condition: Tensor) -> Dict[str, Tensor]:
        embeddings = self.hidden_in(condition)

        output = self.model(encoder_hidden_states=embeddings)

        all_layer_state_dict = {}
        shape_state_dict = self.tokenizer.shape_state_dict
        for layer_index in range(self.layer_num):
            layer_output = output[:, layer_index]
            layer_state_dict = self.tokenizer.detokenize(shape_state_dict, layer_output)
            for key, value in layer_state_dict.items():
                new_key = f"{self.prefix}{layer_index}.{key}"
                all_layer_state_dict[new_key] = value

        all_layer_state_dict["scale"] = torch.tensor(self.tokenizer.lora_scale).to(
            device=output.device, dtype=output.dtype
        )

        return all_layer_state_dict
