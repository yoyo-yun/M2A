import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
from torch.nn import Parameter, init, functional as F

@dataclass
class LoraConfig():
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    user: bool = False
    item: bool = False
    user_dim = 768
    item_dim = 768
    lora_type: int = field(default=1, metadata={"help": "setting for cv or mv"})  # 1 for multiview; 1 for coarse view
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})


# rankn + KRonA
class OurLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.lora_config = None
        self.reset_parameters()
        self.regularization_weights = None

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def injecting_parameters(self, config) -> None:
        self.lora_config = config
        self.view_r = 32
        self.shape_view_A = (self.view_r, self.out_features // self.view_r)
        self.shape_view_B = (self.in_features // self.view_r, self.view_r)

        # lora for coarse views
        self.register_module("lora_A", nn.Linear(self.in_features, self.lora_config["r"], bias=False))
        self.register_module("lora_B", nn.Linear(self.lora_config["r"], self.out_features, bias=False))
        init.zeros_(self.lora_B.weight)

        self.scaling = config["lora_alpha"] / config["r"]
        if config["lora_dropout"] > 0.0:
            lora_dropout_layer = nn.Dropout(p=config["lora_dropout"])
        else:
            def lora_dropout_layer(x):
                return x
        self.register_module("lora_dropout", lora_dropout_layer)

        if self.lora_config["lora_type"] == 1:
            self.register_module("lora_UA", nn.Linear(self.in_features, self.lora_config["r"], bias=False))
            self.register_module("lora_UB", nn.Linear(self.lora_config["r"], self.out_features, bias=False))
            init.zeros_(self.lora_UB.weight)

            self.register_module("lora_IA", nn.Linear(self.in_features, self.lora_config["r"], bias=False))
            self.register_module("lora_IB", nn.Linear(self.lora_config["r"], self.out_features, bias=False))
            init.zeros_(self.lora_IB.weight)

            if self.lora_config["user"]:
                self.register_module("lora_U", nn.Linear(self.in_features,
                                                         self.view_r * (self.in_features // self.view_r), bias=False))
                self.register_parameter("lora_VU", Parameter(torch.empty(*self.shape_view_B)))
                init.kaiming_uniform_(self.lora_VU, a=math.sqrt(5))
            else:
                self.register_module("lora_U", None)
            if self.lora_config["item"]:
                self.register_module("lora_I", nn.Linear(self.in_features,
                                                         self.view_r * (self.in_features // self.view_r), bias=False))
                self.register_parameter("lora_VI", Parameter(torch.empty(*self.shape_view_B)))
                init.kaiming_uniform_(self.lora_VI, a=math.sqrt(5))
            else:
                self.register_module("lora_I", None)

    # nodes_hidden_states (User, Item)
    def forward(self, input: torch.Tensor, nodes_hidden_states=None) -> torch.Tensor:
        if self.lora_config is None:
            return F.linear(input, self.weight, self.bias)
        else:
            if self.lora_config["lora_type"] == 1:
                return self.forward_with_node_mv(input, nodes_hidden_states)
            if self.lora_config["lora_type"] == 0:
                return self.forward_with_node_cv(input)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def forward_with_node_mv(self, input: torch.Tensor, nodes_hidden_states = None) -> torch.Tensor:
        bs = input.shape[0]

        # lora
        result = F.linear(input, self.weight, self.bias)
        result_lora = self.lora_B(self.lora_A(self.lora_dropout(input))) * self.scaling
        result += result_lora

        user_hidden_states, item_hidden_states = None, None
        if nodes_hidden_states is not None:
            user_hidden_states, item_hidden_states = nodes_hidden_states
        if user_hidden_states is not None and self.lora_U is not None:
            weight_u = self.lora_U(user_hidden_states).view(bs, *self.shape_view_A)  # (bs, r*d//r)
            # bs, dim_in, dim_out
            weight_u = torch.einsum("bql, nm->bqnlm", weight_u, self.lora_VU).view(bs, self.in_features, self.out_features).contiguous()
            # if self.regularization_weights is None:
            #     self.regularization_weights = weight_u
            result_u = torch.einsum("bsd, bdq->bsq", input, weight_u) # (bs, seq, dim_out)
            result += result_u
        else:  # for only coarse view
            result_lora_U = self.lora_UB(self.lora_UA(self.lora_dropout(input))) * self.scaling
            result += result_lora_U

        if item_hidden_states is not None and self.lora_I is not None:
            weight_i = self.lora_I(item_hidden_states).view(bs, *self.shape_view_A)
            weight_i = torch.einsum("bql, nm->bqnlm", weight_i, self.lora_VI).view(bs, self.in_features,
                                                                                   self.out_features).contiguous()
            result_i = torch.einsum("bsd, bdq->bsq", input, weight_i)  # (bs, seq, dim_out)
            result += result_i
            # for calculating regularization
            # if self.regularization_weights is None:
            #     self.regularization_weights = weight_i
            # else:
            #     self.regularization_weights = self.regularization_weights + weight_i
        else:  # for only coarse view
            result_lora_I = self.lora_IB(self.lora_IA(self.lora_dropout(input))) * self.scaling
            result += result_lora_I
        return result


    def forward_with_node_cv(self, input: torch.Tensor) -> torch.Tensor:
        # lora
        result = F.linear(input, self.weight, self.bias)
        result_lora = self.lora_B(self.lora_A(self.lora_dropout(input))) * self.scaling
        result += result_lora
        return result


    def forward_with_lora(self, input: torch.Tensor, nodes_hidden_states=None) -> torch.Tensor:
        result = F.linear(input, self.weight, self.bias)
        x = self.lora_dropout(input)
        x = self.lora_A(x)
        x = self.lora_B(x)
        x = x * self.scaling
        result += x
        return result

    def forward_with_lora_node(self, input: torch.Tensor, nodes_hidden_states=None) -> torch.Tensor:
        result = F.linear(input, self.weight, self.bias)
        result_lora = self.lora_B(self.lora_dropout(self.lora_A(input))) * self.scaling
        result += result_lora
        user_hidden_states, item_hidden_states = None, None
        if nodes_hidden_states is not None:
            user_hidden_states, item_hidden_states = nodes_hidden_states
        if user_hidden_states is not None and self.lora_U is not None:
            result_u = self.lora_U(user_hidden_states)
            result += result_u
        if item_hidden_states is not None and self.lora_I is not None:
            result_i = self.lora_I(item_hidden_states)
            result += result_i
        return result


class OurConv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)
        self.regularization_weights = None
        self.lora_config = None

    def injecting_parameters(self, config) -> None:
        self.lora_config = config
        self.view_r = 32
        self.shape_view_A = (self.view_r, self.nf // self.view_r)
        self.shape_view_B = (self.nx // self.view_r, self.view_r)

        self.scaling = config["lora_alpha"] / config["r"]
        if config["lora_dropout"] > 0.0:
            lora_dropout_layer = nn.Dropout(p=config["lora_dropout"])
        else:
            def lora_dropout_layer(x):
                return x
        self.register_module("lora_dropout", lora_dropout_layer)

        if self.lora_config["user"]:
            self.register_module("lora_U", nn.Linear(self.nx,
                                                     self.view_r * (self.nx // self.view_r), bias=False))
            self.register_parameter("lora_VU", Parameter(torch.empty(*self.shape_view_A)))
            init.kaiming_uniform_(self.lora_VU, a=math.sqrt(5))
        else:
            self.register_module("lora_U", None)
        if self.lora_config["item"]:
            self.register_module("lora_I", nn.Linear(self.nx,
                                                     self.view_r * (self.nx // self.view_r), bias=False))
            self.register_parameter("lora_VI", Parameter(torch.empty(*self.shape_view_A)))
            init.kaiming_uniform_(self.lora_VI, a=math.sqrt(5))
        else:
            self.register_module("lora_I", None)

        self.register_module("lora_UA", nn.Linear(self.nx, self.lora_config["r"], bias=False))
        self.register_module("lora_UB", nn.Linear(self.lora_config["r"], self.nf, bias=False))
        init.zeros_(self.lora_UB.weight)

        self.register_module("lora_IA", nn.Linear(self.nx, self.lora_config["r"], bias=False))
        self.register_module("lora_IB", nn.Linear(self.lora_config["r"], self.nf, bias=False))
        init.zeros_(self.lora_IB.weight)

    def forward(self, x, nodes_hidden_states: Optional[Tuple[torch.FloatTensor]] = None):
        input = x
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)  # (bs, eq, dim)
        if self.lora_config is None:
            return x
        bs = x.shape[0]
        user_hidden_states, item_hidden_states = None, None
        if nodes_hidden_states is not None:
            user_hidden_states, item_hidden_states = nodes_hidden_states
        if user_hidden_states is not None and self.lora_U is not None:
            weight_u = self.lora_U(user_hidden_states).view(bs, *self.shape_view_B)  # (bs, r*din//r)
            # bs, dim_in, dim_out
            weight_u = torch.einsum("bql, nm->bqnlm", weight_u, self.lora_VU).view(bs, self.nx, self.nf).contiguous()
            result_u = torch.einsum("bsd, bdq->bsq", input, weight_u)  # (bs, seq, dim_out)
            x += result_u
        else:
            result_lora_U = self.lora_UB(self.lora_UA(self.lora_dropout(input))) * self.scaling
            x += result_lora_U
        if item_hidden_states is not None and self.lora_I is not None:
            weight_i = self.lora_I(item_hidden_states).view(bs, *self.shape_view_B)
            weight_i = torch.einsum("bql, nm->bqnlm", weight_i, self.lora_VI).view(bs, self.nx, self.nf).contiguous()
            result_i = torch.einsum("bsd, bdq->bsq", input, weight_i)  # (bs, seq, dim_out)
            x += result_i
        else:
            result_lora_I = self.lora_IB(self.lora_IA(self.lora_dropout(input))) * self.scaling
            x += result_lora_I
        return x