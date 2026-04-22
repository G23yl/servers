from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops import rearrange

from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from examples.depth_wan.layers import ResidualMLP


class TrajectoryFPNHead(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self, in_channels, out_channels=16, num_groups=16, num_tensors=30, patch_size=2
    ):
        """
        Args:
            in_channels (int): Number of input channels for each tensor in the list.
            hidden_dim (int): Hidden dimension of the MLP.
            out_channels (int): Number of output channels for the final tensor. Default is 2.
        """
        super(TrajectoryFPNHead, self).__init__()
        hidden_dim = 64
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, hidden_dim, kernel_size=3, stride=1, padding=1
                    ),
                    nn.GroupNorm(
                        num_groups=min(num_groups, hidden_dim), num_channels=hidden_dim
                    ),
                    nn.ReLU(inplace=True),
                )
                for i in range(num_tensors)
            ]
        )
        self.conv_out = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(
            scale_factor=patch_size, mode="bilinear", align_corners=False
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(self, tensor_list):
        """
        Args:
            tensor_list (list of torch.Tensor): List of tensors with shape [B, C, H, W].

        Returns:
            torch.Tensor: A tensor with shape [B, C_out, H, W].
        """
        if len(tensor_list) != len(self.convs):
            raise ValueError(
                f"Number of input tensors ({len(tensor_list)}) must match the number of conv layers ({len(self.convs)})."
            )

        convolved_tensors = []
        for conv, tensor in zip(self.convs, tensor_list):
            tensor = conv(tensor)
            convolved_tensors.append(tensor)

        summed_tensor = torch.sum(torch.stack(convolved_tensors), dim=0)
        summed_tensor = self.upsample(summed_tensor)
        output = self.conv_out(summed_tensor)
        return output


class AIDN(nn.Module):
    def __init__(self, nhidden=1536, hidden_dim=256, ks=1, time_adaptive=True):
        super().__init__()

        pw = ks // 2
        affine_func = partial(nn.Conv2d, kernel_size=ks, padding=pw)
        self.gamma_mlp = ResidualMLP(
            input_dim=nhidden,
            hidden_dim=hidden_dim,
            output_dim=nhidden,
            num_mlp=1,
            num_layer_per_mlp=3,
            affine_func=affine_func,
        )
        self.beta_mlp = ResidualMLP(
            input_dim=nhidden,
            hidden_dim=hidden_dim,
            output_dim=nhidden,
            num_mlp=1,
            num_layer_per_mlp=3,
            affine_func=affine_func,
        )
        self.time_adaptive_scale = None

        if time_adaptive:
            self.time_adaptive_scale = nn.Sequential(
                nn.Linear(6 * nhidden, 1),
                nn.Sigmoid(),
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.zeros_(module.weight)
                # if module.bias is not None:
                #     nn.init.zeros_(module.bias)

    def forward(self, x, timestep, modal1_feats, modal2_feats=None):
        # [1, 26520, 1536]
        b, l, c = x.shape
        # h = w = int(math.sqrt(l))
        f = 13
        s = l // f
        x = x.reshape(b, f, s, c).permute(0, 3, 1, 2)
        modal1_feats = modal1_feats.reshape(b, f, s, c).permute(0, 3, 1, 2)

        if modal2_feats is not None:
            modal2_feats = modal2_feats.reshape(b, f, s, c).permute(0, 3, 1, 2)

            gamma1, beta1 = self._forward(modal1_feats)
            gamma2, beta2 = self._forward(modal2_feats)

            gamma = (gamma1 + gamma2) / 2
            beta = (beta1 + beta2) / 2
        else:
            gamma = self.gamma_mlp(modal1_feats)
            beta = self.beta_mlp(modal1_feats)

        if self.time_adaptive_scale is not None:
            assert timestep is not None
            sigma = self.time_adaptive_scale(timestep)
            sigma = sigma.unsqueeze(-1).unsqueeze(-1)
            out = x * (1 + sigma * gamma) + sigma * beta
        else:
            out = x * (1 + gamma) + beta

        out = out.permute(0, 2, 3, 1).reshape(b, -1, c)
        return out

    def _forward(self, modal_feats):
        gamma = self.gamma_mlp(modal_feats)
        beta = self.beta_mlp(modal_feats)
        return gamma, beta


class DepthAIDNs(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, num_layers=10, time_adaptive=True) -> None:
        super().__init__()

        self.num_layers = num_layers

        self.D2I_aidn_blocks = nn.ModuleList(
            [AIDN(time_adaptive=time_adaptive) for i in range(num_layers)]
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def train(self, mode: bool = True):
        for block in self.D2I_aidn_blocks:
            block.train(mode)

class ThreeBranchAIDNs(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, num_layers=10, time_adaptive=True) -> None:
        super().__init__()

        self.num_layers = num_layers

        self.D2V_aidn_blocks = nn.ModuleList(
            [AIDN(time_adaptive=time_adaptive) for i in range(num_layers)]
        )
        self.D2M_aidn_blocks = nn.ModuleList(
            [AIDN(time_adaptive=time_adaptive) for i in range(num_layers)]
        )
        self.M2D_aidn_blocks = nn.ModuleList(
            [AIDN(time_adaptive=time_adaptive) for i in range(num_layers)]
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def train(self, mode: bool = True):
        for block in self.D2V_aidn_blocks:
            block.train(mode)
        for block in self.D2M_aidn_blocks:
            block.train(mode)
        for block in self.M2D_aidn_blocks:
            block.train(mode)

class DepthWanModel(nn.Module):
    def __init__(
        self, rgb_transformer: WanModel, depth_transformer: WanModel, aidns: DepthAIDNs
    ):
        super().__init__()
        self.rgb_transformer = rgb_transformer
        self.depth_transformer = depth_transformer
        self.aidns = aidns
        self.target_idxs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

    def _prepare_input(
        self,
        transformer: WanModel,
        timestep: torch.Tensor,
        context: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
    ):
        t = transformer.time_embedding(
            sinusoidal_embedding_1d(transformer.freq_dim, timestep)
        )
        t_mod = transformer.time_projection(t).unflatten(1, (6, transformer.dim))
        context = transformer.text_embedding(context)

        if transformer.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = transformer.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = transformer.patchify(x)

        freqs = (
            torch.cat(
                [
                    transformer.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        return x, context, t, t_mod, freqs, (f, h, w)

    def output(
        self,
        transformer: WanModel,
        hidden_state: torch.Tensor,
        embed_t: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ):
        x = transformer.head(hidden_state, embed_t)
        x = transformer.unpatchify(x, grid_size)
        return x

    def forward(
        self,
        rgb_x: torch.Tensor,
        depth_x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        rgb_clip_feature: Optional[torch.Tensor] = None,
        depth_clip_feature: Optional[torch.Tensor] = None,
        rgb_y: Optional[torch.Tensor] = None,
        depth_y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
    ):
        ## for gradient checkpoint
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        (
            rgb_hidden_state,
            rgb_context,
            rgb_embed_t,
            rgb_t_mod,
            rgb_freqs,
            rgb_grid_size,
        ) = self._prepare_input(
            self.rgb_transformer, timestep, context, rgb_x, rgb_y, rgb_clip_feature
        )
        (
            depth_hidden_state,
            depth_context,
            depth_embed_t,
            depth_t_mod,
            depth_freqs,
            depth_grid_size,
        ) = self._prepare_input(
            self.depth_transformer,
            timestep,
            context,
            depth_x,
            depth_y,
            depth_clip_feature,
        )

        for block_idx in range(len(self.rgb_transformer.blocks)):
            ## rgb 更新一次
            rgb_block = self.rgb_transformer.blocks[block_idx]
            if use_gradient_checkpointing and self.rgb_transformer.training:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        rgb_hidden_state, rgb_cross_attn_map = torch_checkpoint(
                            create_custom_forward(rgb_block),
                            rgb_hidden_state,
                            rgb_context,
                            rgb_t_mod,
                            rgb_freqs,
                            use_reentrant=False,
                        )
                else:
                    rgb_hidden_state, rgb_cross_attn_map = torch_checkpoint(
                        create_custom_forward(rgb_block),
                        rgb_hidden_state,
                        rgb_context,
                        rgb_t_mod,
                        rgb_freqs,
                        use_reentrant=False,
                    )
            else:
                rgb_hidden_state, rgb_cross_attn_map = rgb_block(
                    rgb_hidden_state, rgb_context, rgb_t_mod, rgb_freqs
                )

            ## 更新depth一次
            depth_block = self.depth_transformer.blocks[block_idx]
            if use_gradient_checkpointing and self.depth_transformer.training:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        depth_hidden_state, _ = torch_checkpoint(
                            create_custom_forward(depth_block),
                            depth_hidden_state,
                            depth_context,
                            depth_t_mod,
                            depth_freqs,
                            rgb_cross_attn_map,
                            use_reentrant=False,
                        )
                else:
                    depth_hidden_state, _ = torch_checkpoint(
                        create_custom_forward(depth_block),
                        depth_hidden_state,
                        depth_context,
                        depth_t_mod,
                        depth_freqs,
                        rgb_cross_attn_map,
                        use_reentrant=False,
                    )
            else:
                depth_hidden_state, _ = depth_block(
                    depth_hidden_state,
                    depth_context,
                    depth_t_mod,
                    depth_freqs,
                    rgb_cross_attn_map,
                )

            ## depth to image
            if block_idx in self.target_idxs:
                id = self.target_idxs.index(block_idx)
                aidn_block = self.aidns.D2I_aidn_blocks[id]
                if use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            rgb_hidden_state = torch_checkpoint(
                                create_custom_forward(aidn_block),
                                rgb_hidden_state,
                                rgb_t_mod.flatten(1),
                                depth_hidden_state,
                                use_reentrant=False,
                            )
                    else:
                        rgb_hidden_state = torch_checkpoint(
                            create_custom_forward(aidn_block),
                            rgb_hidden_state,
                            rgb_t_mod.flatten(1),
                            depth_hidden_state,
                            use_reentrant=False,
                        )
                else:
                    rgb_hidden_state = aidn_block(
                        rgb_hidden_state, rgb_t_mod.flatten(1), depth_hidden_state
                    )

        rgb_out = self.output(
            self.rgb_transformer, rgb_hidden_state, rgb_embed_t, rgb_grid_size
        )
        depth_out = self.output(
            self.depth_transformer, depth_hidden_state, depth_embed_t, depth_grid_size
        )
        return rgb_out, depth_out

class DepthWanThreeBranchModel(nn.Module):
    def __init__(
        self, rgb_transformer: WanModel, depth_transformer: WanModel, mask_transformer: WanModel, aidns: ThreeBranchAIDNs
    ):
        super().__init__()
        self.rgb_transformer = rgb_transformer
        self.depth_transformer = depth_transformer
        self.mask_transformer = mask_transformer
        self.aidns = aidns
        self.target_idxs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

    def _prepare_input(
        self,
        transformer: WanModel,
        timestep: torch.Tensor,
        context: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
    ):
        t = transformer.time_embedding(
            sinusoidal_embedding_1d(transformer.freq_dim, timestep)
        )
        t_mod = transformer.time_projection(t).unflatten(1, (6, transformer.dim))
        context = transformer.text_embedding(context)

        if transformer.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = transformer.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = transformer.patchify(x)

        freqs = (
            torch.cat(
                [
                    transformer.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        return x, context, t, t_mod, freqs, (f, h, w)

    def output(
        self,
        transformer: WanModel,
        hidden_state: torch.Tensor,
        embed_t: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ):
        x = transformer.head(hidden_state, embed_t)
        x = transformer.unpatchify(x, grid_size)
        return x

    def forward(
        self,
        rgb_x: torch.Tensor,
        depth_x: torch.Tensor,
        mask_x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        rgb_clip_feature: Optional[torch.Tensor] = None,
        depth_clip_feature: Optional[torch.Tensor] = None,
        mask_clip_feature: Optional[torch.Tensor] = None,
        rgb_y: Optional[torch.Tensor] = None,
        depth_y: Optional[torch.Tensor] = None,
        mask_y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
    ):
        ## for gradient checkpoint
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        (
            rgb_hidden_state,
            rgb_context,
            rgb_embed_t,
            rgb_t_mod,
            rgb_freqs,
            rgb_grid_size,
        ) = self._prepare_input(
            self.rgb_transformer, timestep, context, rgb_x, rgb_y, rgb_clip_feature
        )
        (
            depth_hidden_state,
            depth_context,
            depth_embed_t,
            depth_t_mod,
            depth_freqs,
            depth_grid_size,
        ) = self._prepare_input(
            self.depth_transformer,
            timestep,
            context,
            depth_x,
            depth_y,
            depth_clip_feature,
        )
        (
            mask_hidden_state,
            mask_context,
            mask_embed_t,
            mask_t_mod,
            mask_freqs,
            mask_grid_size,
        ) = self._prepare_input(
            self.mask_transformer,
            timestep,
            context,
            mask_x,
            mask_y,
            mask_clip_feature,
        )

        for block_idx in range(len(self.rgb_transformer.blocks)):
            ## rgb 更新一次
            rgb_block = self.rgb_transformer.blocks[block_idx]
            if use_gradient_checkpointing and self.rgb_transformer.training:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        rgb_hidden_state, rgb_cross_attn_map = torch_checkpoint(
                            create_custom_forward(rgb_block),
                            rgb_hidden_state,
                            rgb_context,
                            rgb_t_mod,
                            rgb_freqs,
                            use_reentrant=False,
                        )
                else:
                    rgb_hidden_state, rgb_cross_attn_map = torch_checkpoint(
                        create_custom_forward(rgb_block),
                        rgb_hidden_state,
                        rgb_context,
                        rgb_t_mod,
                        rgb_freqs,
                        use_reentrant=False,
                    )
            else:
                rgb_hidden_state, rgb_cross_attn_map = rgb_block(
                    rgb_hidden_state, rgb_context, rgb_t_mod, rgb_freqs
                )

            if block_idx in self.target_idxs:
                # depth2mask
                id = self.target_idxs.index(block_idx)
                aidn_block = self.aidns.D2M_aidn_blocks[id]
                if use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            mask_hidden_state = torch_checkpoint(
                                create_custom_forward(aidn_block),
                                mask_hidden_state,
                                mask_t_mod.flatten(1),
                                depth_hidden_state,
                                use_reentrant=False,
                            )
                    else:
                        mask_hidden_state = torch_checkpoint(
                            create_custom_forward(aidn_block),
                            mask_hidden_state,
                            mask_t_mod.flatten(1),
                            depth_hidden_state,
                            use_reentrant=False,
                        )
                else:
                    mask_hidden_state = aidn_block(
                        mask_hidden_state, mask_t_mod.flatten(1), depth_hidden_state
                    )

            ## 更新depth一次
            depth_block = self.depth_transformer.blocks[block_idx]
            if use_gradient_checkpointing and self.depth_transformer.training:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        depth_hidden_state, _ = torch_checkpoint(
                            create_custom_forward(depth_block),
                            depth_hidden_state,
                            depth_context,
                            depth_t_mod,
                            depth_freqs,
                            rgb_cross_attn_map,
                            use_reentrant=False,
                        )
                else:
                    depth_hidden_state, _ = torch_checkpoint(
                        create_custom_forward(depth_block),
                        depth_hidden_state,
                        depth_context,
                        depth_t_mod,
                        depth_freqs,
                        rgb_cross_attn_map,
                        use_reentrant=False,
                    )
            else:
                depth_hidden_state, _ = depth_block(
                    depth_hidden_state,
                    depth_context,
                    depth_t_mod,
                    depth_freqs,
                    rgb_cross_attn_map,
                )

            ## 更新mask一次
            mask_block = self.mask_transformer.blocks[block_idx]
            if use_gradient_checkpointing and self.mask_transformer.training:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        mask_hidden_state, _ = torch_checkpoint(
                            create_custom_forward(mask_block),
                            mask_hidden_state,
                            mask_context,
                            mask_t_mod,
                            mask_freqs,
                            rgb_cross_attn_map,
                            use_reentrant=False,
                        )
                else:
                    mask_hidden_state, _ = torch_checkpoint(
                        create_custom_forward(mask_block),
                        mask_hidden_state,
                        mask_context,
                        mask_t_mod,
                        mask_freqs,
                        rgb_cross_attn_map,
                        use_reentrant=False,
                    )
            else:
                mask_hidden_state, _ = mask_block(
                    mask_hidden_state,
                    mask_context,
                    mask_t_mod,
                    mask_freqs,
                    rgb_cross_attn_map,
                )

            if block_idx in self.target_idxs:
                # mask2depth
                id = self.target_idxs.index(block_idx)
                aidn_block = self.aidns.M2D_aidn_blocks[id]
                if use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            depth_hidden_state = torch_checkpoint(
                                create_custom_forward(aidn_block),
                                depth_hidden_state,
                                depth_t_mod.flatten(1),
                                mask_hidden_state,
                                use_reentrant=False,
                            )
                    else:
                        depth_hidden_state = torch_checkpoint(
                            create_custom_forward(aidn_block),
                            depth_hidden_state,
                            depth_t_mod.flatten(1),
                            mask_hidden_state,
                            use_reentrant=False,
                        )
                else:
                    depth_hidden_state = aidn_block(
                        depth_hidden_state, depth_t_mod.flatten(1), mask_hidden_state
                    )

            ## depth to image
            if block_idx in self.target_idxs:
                id = self.target_idxs.index(block_idx)
                aidn_block = self.aidns.D2V_aidn_blocks[id]
                if use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            rgb_hidden_state = torch_checkpoint(
                                create_custom_forward(aidn_block),
                                rgb_hidden_state,
                                rgb_t_mod.flatten(1),
                                depth_hidden_state,
                                use_reentrant=False,
                            )
                    else:
                        rgb_hidden_state = torch_checkpoint(
                            create_custom_forward(aidn_block),
                            rgb_hidden_state,
                            rgb_t_mod.flatten(1),
                            depth_hidden_state,
                            use_reentrant=False,
                        )
                else:
                    rgb_hidden_state = aidn_block(
                        rgb_hidden_state, rgb_t_mod.flatten(1), depth_hidden_state
                    )

        rgb_out = self.output(
            self.rgb_transformer, rgb_hidden_state, rgb_embed_t, rgb_grid_size
        )
        depth_out = self.output(
            self.depth_transformer, depth_hidden_state, depth_embed_t, depth_grid_size
        )
        mask_out = self.output(
            self.mask_transformer, mask_hidden_state, mask_embed_t, mask_grid_size
        )
        return rgb_out, depth_out, mask_out

class DepthWanFullModel(nn.Module):
    def __init__(
        self, rgb_transformer: WanModel, depth_transformer: WanModel, aidns: DepthAIDNs
    ):
        super().__init__()
        self.rgb_transformer = rgb_transformer
        self.depth_transformer = depth_transformer
        self.aidns = aidns
        self.target_idxs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

    def _prepare_input(
        self,
        transformer: WanModel,
        timestep: torch.Tensor,
        context: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
    ):
        t = transformer.time_embedding(
            sinusoidal_embedding_1d(transformer.freq_dim, timestep)
        )
        t_mod = transformer.time_projection(t).unflatten(1, (6, transformer.dim))
        context = transformer.text_embedding(context)

        if transformer.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = transformer.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = transformer.patchify(x)

        freqs = (
            torch.cat(
                [
                    transformer.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        return x, context, t, t_mod, freqs, (f, h, w)

    def output(
        self,
        transformer: WanModel,
        hidden_state: torch.Tensor,
        embed_t: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ):
        x = transformer.head(hidden_state, embed_t)
        x = transformer.unpatchify(x, grid_size)
        return x

    def forward(
        self,
        rgb_x: torch.Tensor,
        depth_x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        rgb_clip_feature: Optional[torch.Tensor] = None,
        depth_clip_feature: Optional[torch.Tensor] = None,
        rgb_y: Optional[torch.Tensor] = None,
        depth_y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
    ):
        ## for gradient checkpoint
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        (
            rgb_hidden_state,
            rgb_context,
            rgb_embed_t,
            rgb_t_mod,
            rgb_freqs,
            rgb_grid_size,
        ) = self._prepare_input(
            self.rgb_transformer, timestep, context, rgb_x, rgb_y, rgb_clip_feature
        )
        (
            depth_hidden_state,
            depth_context,
            depth_embed_t,
            depth_t_mod,
            depth_freqs,
            depth_grid_size,
        ) = self._prepare_input(
            self.depth_transformer,
            timestep,
            context,
            depth_x,
            depth_y,
            depth_clip_feature,
        )

        for block_idx in range(len(self.rgb_transformer.blocks)):
            ## rgb 更新一次
            rgb_block = self.rgb_transformer.blocks[block_idx]
            if use_gradient_checkpointing and self.rgb_transformer.training:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        rgb_hidden_state, rgb_cross_attn_map = torch_checkpoint(
                            create_custom_forward(rgb_block),
                            rgb_hidden_state,
                            rgb_context,
                            rgb_t_mod,
                            rgb_freqs,
                            use_reentrant=False,
                        )
                else:
                    rgb_hidden_state, rgb_cross_attn_map = torch_checkpoint(
                        create_custom_forward(rgb_block),
                        rgb_hidden_state,
                        rgb_context,
                        rgb_t_mod,
                        rgb_freqs,
                        use_reentrant=False,
                    )
            else:
                rgb_hidden_state, rgb_cross_attn_map = rgb_block(
                    rgb_hidden_state, rgb_context, rgb_t_mod, rgb_freqs
                )

            ## 更新depth一次
            depth_block = self.depth_transformer.blocks[block_idx]
            if use_gradient_checkpointing and self.depth_transformer.training:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        depth_hidden_state, _ = torch_checkpoint(
                            create_custom_forward(depth_block),
                            depth_hidden_state,
                            depth_context,
                            depth_t_mod,
                            depth_freqs,
                            rgb_cross_attn_map,
                            use_reentrant=False,
                        )
                else:
                    depth_hidden_state, _ = torch_checkpoint(
                        create_custom_forward(depth_block),
                        depth_hidden_state,
                        depth_context,
                        depth_t_mod,
                        depth_freqs,
                        rgb_cross_attn_map,
                        use_reentrant=False,
                    )
            else:
                depth_hidden_state, _ = depth_block(
                    depth_hidden_state,
                    depth_context,
                    depth_t_mod,
                    depth_freqs,
                    rgb_cross_attn_map,
                )

            if block_idx in self.target_idxs:
                ## depth to image
                id = self.target_idxs.index(block_idx)
                aidn_block = self.aidns.D2I_aidn_blocks[id]
                if use_gradient_checkpointing:
                    if use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            rgb_hidden_state = torch_checkpoint(
                                create_custom_forward(aidn_block),
                                rgb_hidden_state,
                                rgb_t_mod.flatten(1),
                                depth_hidden_state,
                                use_reentrant=False,
                            )
                    else:
                        rgb_hidden_state = torch_checkpoint(
                            create_custom_forward(aidn_block),
                            rgb_hidden_state,
                            rgb_t_mod.flatten(1),
                            depth_hidden_state,
                            use_reentrant=False,
                        )
                else:
                    rgb_hidden_state = aidn_block(
                        rgb_hidden_state, rgb_t_mod.flatten(1), depth_hidden_state
                    )

        rgb_out = self.output(
            self.rgb_transformer, rgb_hidden_state, rgb_embed_t, rgb_grid_size
        )
        depth_out = self.output(
            self.depth_transformer, depth_hidden_state, depth_embed_t, depth_grid_size
        )
        return rgb_out, depth_out

class DepthWanTryModel1(nn.Module):
    def __init__(
        self,
        rgb_transformer: WanModel,
        depth_transformer: WanModel,
        aidns: DepthAIDNs,
        trajectory_head: TrajectoryFPNHead | None = None,
    ):
        super().__init__()
        self.rgb_transformer = rgb_transformer
        self.depth_transformer = depth_transformer
        self.aidns = aidns
        self.target_idxs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        self.trajectory_head = trajectory_head

    def _prepare_input(
        self,
        transformer: WanModel,
        timestep: torch.Tensor,
        context: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
    ):
        t = transformer.time_embedding(
            sinusoidal_embedding_1d(transformer.freq_dim, timestep)
        )
        t_mod = transformer.time_projection(t).unflatten(1, (6, transformer.dim))
        context = transformer.text_embedding(context)

        if transformer.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = transformer.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = transformer.patchify(x)

        freqs = (
            torch.cat(
                [
                    transformer.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        return x, context, t, t_mod, freqs, (f, h, w)

    def output(
        self,
        transformer: WanModel,
        hidden_state: torch.Tensor,
        embed_t: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ):
        x = transformer.head(hidden_state, embed_t)
        x = transformer.unpatchify(x, grid_size)
        return x

    def forward(
        self,
        rgb_x: torch.Tensor,
        depth_x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        rgb_clip_feature: Optional[torch.Tensor] = None,
        depth_clip_feature: Optional[torch.Tensor] = None,
        rgb_y: Optional[torch.Tensor] = None,
        depth_y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
    ):
        ## for gradient checkpoint
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        (
            rgb_hidden_state,
            rgb_context,
            rgb_embed_t,
            rgb_t_mod,
            rgb_freqs,
            rgb_grid_size,
        ) = self._prepare_input(
            self.rgb_transformer, timestep, context, rgb_x, rgb_y, rgb_clip_feature
        )
        (
            depth_hidden_state,
            depth_context,
            depth_embed_t,
            depth_t_mod,
            depth_freqs,
            depth_grid_size,
        ) = self._prepare_input(
            self.depth_transformer,
            timestep,
            context,
            depth_x,
            depth_y,
            depth_clip_feature,
        )

        diff_features = []
        for block_idx in range(len(self.rgb_transformer.blocks)):
            ## rgb 更新一次
            rgb_block = self.rgb_transformer.blocks[block_idx]
            if use_gradient_checkpointing and self.rgb_transformer.training:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        rgb_hidden_state, rgb_cross_attn_map = torch_checkpoint(
                            create_custom_forward(rgb_block),
                            rgb_hidden_state,
                            rgb_context,
                            rgb_t_mod,
                            rgb_freqs,
                            use_reentrant=False,
                        )
                else:
                    rgb_hidden_state, rgb_cross_attn_map = torch_checkpoint(
                        create_custom_forward(rgb_block),
                        rgb_hidden_state,
                        rgb_context,
                        rgb_t_mod,
                        rgb_freqs,
                        use_reentrant=False,
                    )
            else:
                rgb_hidden_state, rgb_cross_attn_map = rgb_block(
                    rgb_hidden_state, rgb_context, rgb_t_mod, rgb_freqs
                )

            ## image to depth
            if block_idx in self.target_idxs:
                id = self.target_idxs.index(block_idx)
                depth_hidden_state = self.aidns.D2I_aidn_blocks[id](
                    depth_hidden_state, depth_t_mod.flatten(1), rgb_hidden_state
                )

            ## 更新depth一次
            depth_block = self.depth_transformer.blocks[block_idx]
            if use_gradient_checkpointing and self.depth_transformer.training:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        depth_hidden_state, _ = torch_checkpoint(
                            create_custom_forward(depth_block),
                            depth_hidden_state,
                            depth_context,
                            depth_t_mod,
                            depth_freqs,
                            rgb_cross_attn_map,
                            use_reentrant=False,
                        )
                else:
                    depth_hidden_state, _ = torch_checkpoint(
                        create_custom_forward(depth_block),
                        depth_hidden_state,
                        depth_context,
                        depth_t_mod,
                        depth_freqs,
                        rgb_cross_attn_map,
                        use_reentrant=False,
                    )
            else:
                depth_hidden_state, _ = depth_block(
                    depth_hidden_state,
                    depth_context,
                    depth_t_mod,
                    depth_freqs,
                    rgb_cross_attn_map,
                )

            diff_features.append(rgb_hidden_state)

        # diff_features dim: [N, B, (F H W), C]
        if self.trajectory_head is not None:
            # features dim: [N, B, C,, H, W]
            features = []
            for i in range(len(diff_features)):
                t_feature = rearrange(
                    diff_features[i],
                    "B (F H W) C -> (B F) C H W",
                    F=rgb_grid_size[0],
                    H=rgb_grid_size[1],
                    W=rgb_grid_size[2],
                )
                features.append(t_feature)
            trajectory_out = self.trajectory_head(features)
            trajectory_out = rearrange(
                trajectory_out, "(B F) C H W -> B C F H W", F=rgb_grid_size[0]
            )

        rgb_out = self.output(
            self.rgb_transformer, rgb_hidden_state, rgb_embed_t, rgb_grid_size
        )
        depth_out = self.output(
            self.depth_transformer, depth_hidden_state, depth_embed_t, depth_grid_size
        )

        if self.trajectory_head is not None:
            return rgb_out, depth_out, trajectory_out

        return rgb_out, depth_out


class DepthWanTrajModel(nn.Module):
    def __init__(
        self,
        rgb_transformer: WanModel,
        depth_transformer: WanModel,
        aidns: DepthAIDNs,
        trajectory_head: TrajectoryFPNHead,
    ):
        super().__init__()
        self.rgb_transformer = rgb_transformer
        self.depth_transformer = depth_transformer
        self.aidns = aidns
        self.target_idxs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        self.trajectory_head = trajectory_head

    def _prepare_input(
        self,
        transformer: WanModel,
        timestep: torch.Tensor,
        context: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
    ):
        t = transformer.time_embedding(
            sinusoidal_embedding_1d(transformer.freq_dim, timestep)
        )
        t_mod = transformer.time_projection(t).unflatten(1, (6, transformer.dim))
        context = transformer.text_embedding(context)

        if transformer.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = transformer.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = transformer.patchify(x)

        freqs = (
            torch.cat(
                [
                    transformer.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        return x, context, t, t_mod, freqs, (f, h, w)

    def output(
        self,
        transformer: WanModel,
        hidden_state: torch.Tensor,
        embed_t: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ):
        x = transformer.head(hidden_state, embed_t)
        x = transformer.unpatchify(x, grid_size)
        return x

    def forward(
        self,
        rgb_x: torch.Tensor,
        depth_x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        rgb_clip_feature: Optional[torch.Tensor] = None,
        depth_clip_feature: Optional[torch.Tensor] = None,
        rgb_y: Optional[torch.Tensor] = None,
        depth_y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
    ):
        ## for gradient checkpoint
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        (
            rgb_hidden_state,
            rgb_context,
            rgb_embed_t,
            rgb_t_mod,
            rgb_freqs,
            rgb_grid_size,
        ) = self._prepare_input(
            self.rgb_transformer, timestep, context, rgb_x, rgb_y, rgb_clip_feature
        )
        (
            depth_hidden_state,
            depth_context,
            depth_embed_t,
            depth_t_mod,
            depth_freqs,
            depth_grid_size,
        ) = self._prepare_input(
            self.depth_transformer,
            timestep,
            context,
            depth_x,
            depth_y,
            depth_clip_feature,
        )

        diff_features = []
        for block_idx in range(len(self.rgb_transformer.blocks)):
            ## rgb 更新一次
            rgb_block = self.rgb_transformer.blocks[block_idx]
            if use_gradient_checkpointing and self.rgb_transformer.training:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        rgb_hidden_state, rgb_cross_attn_map = torch_checkpoint(
                            create_custom_forward(rgb_block),
                            rgb_hidden_state,
                            rgb_context,
                            rgb_t_mod,
                            rgb_freqs,
                            use_reentrant=False,
                        )
                else:
                    rgb_hidden_state, rgb_cross_attn_map = torch_checkpoint(
                        create_custom_forward(rgb_block),
                        rgb_hidden_state,
                        rgb_context,
                        rgb_t_mod,
                        rgb_freqs,
                        use_reentrant=False,
                    )
            else:
                rgb_hidden_state, rgb_cross_attn_map = rgb_block(
                    rgb_hidden_state, rgb_context, rgb_t_mod, rgb_freqs
                )

            ## 更新depth一次
            depth_block = self.depth_transformer.blocks[block_idx]
            if use_gradient_checkpointing and self.depth_transformer.training:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        depth_hidden_state, _ = torch_checkpoint(
                            create_custom_forward(depth_block),
                            depth_hidden_state,
                            depth_context,
                            depth_t_mod,
                            depth_freqs,
                            rgb_cross_attn_map,
                            use_reentrant=False,
                        )
                else:
                    depth_hidden_state, _ = torch_checkpoint(
                        create_custom_forward(depth_block),
                        depth_hidden_state,
                        depth_context,
                        depth_t_mod,
                        depth_freqs,
                        rgb_cross_attn_map,
                        use_reentrant=False,
                    )
            else:
                depth_hidden_state, _ = depth_block(
                    depth_hidden_state,
                    depth_context,
                    depth_t_mod,
                    depth_freqs,
                    rgb_cross_attn_map,
                )

            ## depth to image
            if block_idx in self.target_idxs:
                id = self.target_idxs.index(block_idx)
                rgb_hidden_state = self.aidns.D2I_aidn_blocks[id](
                    rgb_hidden_state, rgb_t_mod.flatten(1), depth_hidden_state
                )

            diff_features.append(rgb_hidden_state)

        # diff_features dim: [N, B, (F H W), C]
        # features dim: [N, B, C,, H, W]
        features = []
        for i in range(len(diff_features)):
            t_feature = rearrange(
                diff_features[i],
                "B (F H W) C -> (B F) C H W",
                F=rgb_grid_size[0],
                H=rgb_grid_size[1],
                W=rgb_grid_size[2],
            )
            features.append(t_feature)
        trajectory_out = self.trajectory_head(features)
        trajectory_out = rearrange(
            trajectory_out, "(B F) C H W -> B C F H W", F=rgb_grid_size[0]
        )

        rgb_out = self.output(
            self.rgb_transformer, rgb_hidden_state, rgb_embed_t, rgb_grid_size
        )
        depth_out = self.output(
            self.depth_transformer, depth_hidden_state, depth_embed_t, depth_grid_size
        )

        return rgb_out, depth_out, trajectory_out


class DepthWanModelNoTAN(nn.Module):
    def __init__(self, rgb_transformer: WanModel, depth_transformer: WanModel):
        super().__init__()
        self.rgb_transformer = rgb_transformer
        self.depth_transformer = depth_transformer
        self.target_idxs = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

    def _prepare_input(
        self,
        transformer: WanModel,
        timestep: torch.Tensor,
        context: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
    ):
        t = transformer.time_embedding(
            sinusoidal_embedding_1d(transformer.freq_dim, timestep)
        )
        t_mod = transformer.time_projection(t).unflatten(1, (6, transformer.dim))
        context = transformer.text_embedding(context)

        if transformer.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = transformer.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)

        x, (f, h, w) = transformer.patchify(x)

        freqs = (
            torch.cat(
                [
                    transformer.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    transformer.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            )
            .reshape(f * h * w, 1, -1)
            .to(x.device)
        )
        return x, context, t, t_mod, freqs, (f, h, w)

    def output(
        self,
        transformer: WanModel,
        hidden_state: torch.Tensor,
        embed_t: torch.Tensor,
        grid_size: Tuple[int, int, int],
    ):
        x = transformer.head(hidden_state, embed_t)
        x = transformer.unpatchify(x, grid_size)
        return x

    def forward(
        self,
        rgb_x: torch.Tensor,
        depth_x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        rgb_clip_feature: Optional[torch.Tensor] = None,
        depth_clip_feature: Optional[torch.Tensor] = None,
        rgb_y: Optional[torch.Tensor] = None,
        depth_y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
    ):
        ## for gradient checkpoint
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        (
            rgb_hidden_state,
            rgb_context,
            rgb_embed_t,
            rgb_t_mod,
            rgb_freqs,
            rgb_grid_size,
        ) = self._prepare_input(
            self.rgb_transformer, timestep, context, rgb_x, rgb_y, rgb_clip_feature
        )
        (
            depth_hidden_state,
            depth_context,
            depth_embed_t,
            depth_t_mod,
            depth_freqs,
            depth_grid_size,
        ) = self._prepare_input(
            self.depth_transformer,
            timestep,
            context,
            depth_x,
            depth_y,
            depth_clip_feature,
        )

        for block_idx in range(len(self.rgb_transformer.blocks)):
            ## rgb 更新一次
            rgb_block = self.rgb_transformer.blocks[block_idx]
            if use_gradient_checkpointing and self.rgb_transformer.training:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        rgb_hidden_state, rgb_cross_attn_map = torch_checkpoint(
                            create_custom_forward(rgb_block),
                            rgb_hidden_state,
                            rgb_context,
                            rgb_t_mod,
                            rgb_freqs,
                            use_reentrant=False,
                        )
                else:
                    rgb_hidden_state, rgb_cross_attn_map = torch_checkpoint(
                        create_custom_forward(rgb_block),
                        rgb_hidden_state,
                        rgb_context,
                        rgb_t_mod,
                        rgb_freqs,
                        use_reentrant=False,
                    )
            else:
                rgb_hidden_state, rgb_cross_attn_map = rgb_block(
                    rgb_hidden_state, rgb_context, rgb_t_mod, rgb_freqs
                )

            ## 更新depth一次
            depth_block = self.depth_transformer.blocks[block_idx]
            if use_gradient_checkpointing and self.depth_transformer.training:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        depth_hidden_state, _ = torch_checkpoint(
                            create_custom_forward(depth_block),
                            depth_hidden_state,
                            depth_context,
                            depth_t_mod,
                            depth_freqs,
                            rgb_cross_attn_map,
                            use_reentrant=False,
                        )
                else:
                    depth_hidden_state, _ = torch_checkpoint(
                        create_custom_forward(depth_block),
                        depth_hidden_state,
                        depth_context,
                        depth_t_mod,
                        depth_freqs,
                        rgb_cross_attn_map,
                        use_reentrant=False,
                    )
            else:
                depth_hidden_state, _ = depth_block(
                    depth_hidden_state,
                    depth_context,
                    depth_t_mod,
                    depth_freqs,
                    rgb_cross_attn_map,
                )

        rgb_out = self.output(
            self.rgb_transformer, rgb_hidden_state, rgb_embed_t, rgb_grid_size
        )
        depth_out = self.output(
            self.depth_transformer, depth_hidden_state, depth_embed_t, depth_grid_size
        )
        return rgb_out, depth_out
