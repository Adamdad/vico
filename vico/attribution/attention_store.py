import numpy as np
import torch
from typing import List
import torch.nn.functional as F
from .op import (
    batch_min_max_matrix_multiplication,
    batch_soft_min_max_matrix_multiplication,
)


class AttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down": [], "mid": [], "up": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, num_heads: int):
        if self.cur_att_layer >= 0 and is_cross:
            if attn.shape[1] == np.prod(self.attn_res):
                self.step_store[place_in_unet].append((attn, num_heads))

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def aggregate_attention(self, from_where: List[str]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()

        for location in from_where:
            for item, num_heads in attention_maps[location]:

                cross_maps = item.reshape(
                    -1, num_heads, self.attn_res[0], self.attn_res[1], item.shape[-1]
                )
                cross_maps = cross_maps.permute(1, 0, 2, 3, 4)
                out.append(cross_maps.detach().cpu())

        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, attn_res):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res


class AttentionVisStore(AttentionStore):
    def __init__(self, attn_res, image_size):
        super().__init__(attn_res)
        self.image_size = image_size
        self.enable_vis = True

    def __call__(
        self, attn: torch.Tensor, is_cross: bool, place_in_unet: str, num_heads: int
    ):
        # print("is_cross", is_cross, "place_in_unet", place_in_unet, attn.shape)
        if self.enable_vis:
            if self.cur_att_layer >= 0 and is_cross:
                self.step_store[place_in_unet].append((attn, num_heads))

            self.cur_att_layer += 1
            if self.cur_att_layer == self.num_att_layers:
                self.cur_att_layer = 0
                self.between_steps()

    def aggregate_attention(self, from_where: List[str]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item, num_heads in attention_maps[location]:

                # Calculate the downsample rate based on the original and current attention map size
                downsample_rate = np.sqrt(
                    self.image_size[0] * self.image_size[1] / item.shape[1]
                )
                attn_res = (
                    int(self.image_size[0] // downsample_rate),
                    int(self.image_size[1] // downsample_rate),
                )

                # Reshape and permute attention maps to prepare for interpolation
                cross_maps = item.reshape(
                    -1, attn_res[0], attn_res[1], item.shape[-1]
                ).permute(0, 3, 1, 2)

                # Interpolate the reshaped attention maps to the original image size
                cross_maps = F.interpolate(
                    cross_maps,
                    size=self.image_size,
                    mode="bicubic",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

                # Reshape to separate out num_heads and permute to bring num_heads to the front
                cross_maps = cross_maps.reshape(
                    -1,
                    num_heads,
                    self.image_size[0],
                    self.image_size[1],
                    item.shape[-1],
                ).permute(1, 0, 2, 3, 4)

                # Compute the mean across heads, add a new batch dimension, and store in output list
                cross_maps = cross_maps.mean(0).cpu()
                out.append(cross_maps.unsqueeze(0))

        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out.numpy()


class AttentionSTVisStoreBase(AttentionVisStore):
    is_animate_diff = False

    @staticmethod
    def get_empty_store():
        return {
            "down": {"spatial": [], "temporal": []},
            "mid": {"spatial": [], "temporal": []},
            "up": {"spatial": [], "temporal": []},
        }

    def __call__(
        self,
        attn: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
        num_heads: int,
        mode="spatial",
    ):
        # print("is_cross", is_cross, "place_in_unet", place_in_unet, attn.shape)
        if mode == "spatial":
            if self.cur_att_layer >= 0 and is_cross:
                self.step_store[place_in_unet]["spatial"].append((attn, num_heads))
        else:
            if self.cur_att_layer >= 0 and not is_cross:
                self.step_store[place_in_unet]["temporal"].append((attn, num_heads))

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()


class AttentionRolloutVisStore(AttentionSTVisStoreBase):

    def __init__(self, attn_res, image_size):
        super().__init__(attn_res, image_size)
        self.text_attention = None

    def preprocess_attention_rollout(self, attention_maps):
        cross_map, temp_map1, temp_map2 = attention_maps

        # Calculate the downsample rate based on the original and current attention map size
        downsample_rate = np.sqrt(
            self.image_size[0] * self.image_size[1] / cross_map.shape[1]
        )
        attn_res = (
            int(self.image_size[0] // downsample_rate),
            int(self.image_size[1] // downsample_rate),
        )

        # Regularize the self-attention maps with identity matrix to stabilize learning
        identity = torch.eye(
            temp_map2.shape[-1], device=temp_map2.device, dtype=temp_map2.dtype
        ).unsqueeze(0)
        temp_map1 = temp_map1 + identity  # / 2
        temp_map2 = temp_map2 + identity  # / 2

        # Compute the temporal effect by multiplying the regularized self-attention maps
        temporal_effect = torch.bmm(temp_map2, temp_map1)

        # Multiply temporal effect with cross_map to get accumulated relation
        # Adjust cross_map dimensions to match temporal_effect for matrix multiplication
        cross_map = cross_map.permute(1, 0, 2)  # [160, total spatial tokens, 77]
        # print("cross_map", cross_map.shape, "temporal_effect", temporal_effect.shape)
        accumulated_relation = torch.bmm(temporal_effect, cross_map)  # [160, 12, 77]

        if self.text_attention is not None:
            text_attention = self.text_attention.repeat(
                accumulated_relation.shape[0], 1, 1
            )
            accumulated_relation = torch.bmm(accumulated_relation, text_attention)

        # Permute and reshape accumulated_relation to match desired output format
        accumulated_relation = accumulated_relation.permute(1, 0, 2)  # [12, 160, 77]

        accumulated_relation = accumulated_relation.reshape(
            -1, attn_res[0], attn_res[1], accumulated_relation.shape[-1]
        ).permute(0, 3, 1, 2)

        accumulated_relation = F.interpolate(
            accumulated_relation,
            size=self.image_size,
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)

        return accumulated_relation.cpu()

    def aggregate_attention(self, from_where: List[str]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        temp_per_spatial = 2
        attention_maps = self.get_average_attention()
        for location in from_where:
            for sid, (cross_map, num_heads) in enumerate(
                attention_maps[location]["spatial"]
            ):
                # Calculate the downsample rate based on the original and current attention map size
                cross_map = cross_map.reshape(
                    -1, num_heads, cross_map.shape[-2], cross_map.shape[-1]
                )
                cross_map = cross_map.permute(1, 0, 2, 3).mean(0)
                attn_chain = [cross_map]
                for tid in range(temp_per_spatial):

                    temp_map, num_heads = attention_maps[location]["temporal"][
                        sid * temp_per_spatial + tid
                    ]
                    temp_map = temp_map.reshape(
                        -1, num_heads, temp_map.shape[-2], temp_map.shape[-1]
                    )
                    temp_map = temp_map.permute(1, 0, 2, 3).mean(0)
                    attn_chain.append(temp_map)

                attn = self.preprocess_attention_rollout(attn_chain).unsqueeze(0)

                out.append(attn)

        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out.detach().cpu().numpy()


class AttentionFlowVisStore(AttentionSTVisStoreBase):
    def __init__(self, attn_res, image_size, mode="flow_hard"):
        super().__init__(attn_res, image_size)
        self.mode = mode

    def preprocess_attention_flow_soft(self, attention_maps, temperature=0.01):
        cross_map, temp_map1, temp_map2 = attention_maps

        # Calculate the downsample rate based on the original and current attention map size
        downsample_rate = np.sqrt(
            self.image_size[0] * self.image_size[1] / cross_map.shape[1]
        )
        attn_res = (
            int(self.image_size[0] // downsample_rate),
            int(self.image_size[1] // downsample_rate),
        )

        # Regularize the self-attention maps with identity matrix to stabilize learning
        identity = torch.eye(
            temp_map2.shape[-1], device=temp_map2.device, dtype=temp_map2.dtype
        ).unsqueeze(0)
        temp_map1 = temp_map1 + identity  # / 2
        temp_map2 = temp_map2 + identity  # / 2

        # Compute the temporal effect by multiplying the regularized self-attention maps
        temporal_effect = batch_soft_min_max_matrix_multiplication(
            temp_map2, temp_map1, temperature=temperature
        )

        # Multiply temporal effect with cross_map to get accumulated relation
        # Adjust cross_map dimensions to match temporal_effect for matrix multiplication
        cross_map = cross_map.permute(1, 0, 2)  # [160, total spatial tokens, 77]
        # print("cross_map", cross_map.shape, "temporal_effect", temporal_effect.shape)
        accumulated_relation = batch_soft_min_max_matrix_multiplication(
            temporal_effect, cross_map, temperature=temperature
        )  # [160, 12, 77]

        # Permute and reshape accumulated_relation to match desired output format
        accumulated_relation = accumulated_relation.permute(1, 0, 2)  # [12, 160, 77]

        accumulated_relation = accumulated_relation.reshape(
            -1, attn_res[0], attn_res[1], accumulated_relation.shape[-1]
        ).permute(0, 3, 1, 2)

        accumulated_relation = F.interpolate(
            accumulated_relation,
            size=self.image_size,
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)

        return accumulated_relation.cpu()

    def preprocess_attention_flow_hard(self, attention_maps):
        cross_map, temp_map1, temp_map2 = attention_maps

        # Calculate the downsample rate based on the original and current attention map size
        downsample_rate = np.sqrt(
            self.image_size[0] * self.image_size[1] / cross_map.shape[1]
        )
        attn_res = (
            int(self.image_size[0] // downsample_rate),
            int(self.image_size[1] // downsample_rate),
        )

        # Regularize the self-attention maps with identity matrix to stabilize learning
        identity = torch.eye(
            temp_map2.shape[-1], device=temp_map2.device, dtype=temp_map2.dtype
        ).unsqueeze(0)
        temp_map1 = temp_map1 + identity  # / 2
        temp_map2 = temp_map2 + identity  # / 2

        # Compute the temporal effect by multiplying the regularized self-attention maps
        temporal_effect = batch_min_max_matrix_multiplication(temp_map2, temp_map1)

        # Multiply temporal effect with cross_map to get accumulated relation
        # Adjust cross_map dimensions to match temporal_effect for matrix multiplication
        cross_map = cross_map.permute(1, 0, 2)  # [160, total spatial tokens, 77]
        # print("cross_map", cross_map.shape, "temporal_effect", temporal_effect.shape)
        accumulated_relation = batch_min_max_matrix_multiplication(
            temporal_effect, cross_map
        )  # [160, 12, 77]

        # Permute and reshape accumulated_relation to match desired output format
        accumulated_relation = accumulated_relation.permute(1, 0, 2)  # [12, 160, 77]

        accumulated_relation = accumulated_relation.reshape(
            -1, attn_res[0], attn_res[1], accumulated_relation.shape[-1]
        ).permute(0, 3, 1, 2)

        accumulated_relation = F.interpolate(
            accumulated_relation,
            size=self.image_size,
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)

        return accumulated_relation.cpu()

    def aggregate_attention(self, from_where: List[str]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        temp_per_spatial = 2
        attention_maps = self.get_average_attention()
        for location in from_where:
            for sid, (cross_map, num_heads) in enumerate(
                attention_maps[location]["spatial"]
            ):

                # Calculate the downsample rate based on the original and current attention map size
                cross_map = cross_map.reshape(
                    -1, num_heads, cross_map.shape[-2], cross_map.shape[-1]
                )
                cross_map = cross_map.permute(1, 0, 2, 3).mean(0)
                attn_chain = [cross_map]
                for tid in range(temp_per_spatial):

                    temp_map, num_heads = attention_maps[location]["temporal"][
                        sid * temp_per_spatial + tid
                    ]
                    temp_map = temp_map.reshape(
                        -1, num_heads, temp_map.shape[-2], temp_map.shape[-1]
                    )
                    temp_map = temp_map.permute(1, 0, 2, 3).mean(0)
                    attn_chain.append(temp_map)

                if self.mode == "flow_hard":
                    attn = self.preprocess_attention_flow_hard(attn_chain).unsqueeze(0)
                elif self.mode == "flow_soft":
                    attn = self.preprocess_attention_flow_soft(attn_chain).unsqueeze(0)
                else:
                    ValueError("Invalid mode")
                out.append(attn)

        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out.detach().cpu().numpy()


class AttentionSTStore(AttentionStore):
    is_animate_diff = False

    @staticmethod
    def get_empty_store():
        return {
            "down": {"spatial": [], "temporal": []},
            "mid": {"spatial": [], "temporal": []},
            "up": {"spatial": [], "temporal": []},
        }

    def __call__(
        self,
        attn: torch.Tensor,
        is_cross: bool,
        place_in_unet: str,
        num_heads: int,
        mode="spatial",
    ):

        if mode == "spatial":
            if self.cur_att_layer >= 0 and is_cross:
                if attn.shape[1] == np.prod(self.attn_res):
                    self.step_store[place_in_unet]["spatial"].append((attn, num_heads))
        else:
            if self.is_animate_diff:
                if self.cur_att_layer >= 0 and not is_cross:
                    if (
                        attn.shape[1] == self.temporal_res
                        and attn.shape[0] == np.prod(self.attn_res) * 8
                    ):
                        self.step_store[place_in_unet]["temporal"].append(
                            (attn, num_heads)
                        )
            else:
                if self.cur_att_layer >= 0 and not is_cross:
                    if (
                        attn.shape[1] == self.temporal_res
                        and attn.shape[0] == np.prod(self.attn_res) * 20
                    ):
                        self.step_store[place_in_unet]["temporal"].append(
                            (attn, num_heads)
                        )

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def preprocess_attention_rollout(self, attention_maps):

        cross_map, temp_map1, temp_map2 = attention_maps
        # print("cross_map", cross_map.shape, "temp_map1", temp_map1.shape, "temp_map2", temp_map2.shape)

        # Regularize the self-attention maps with identity matrix to stabilize learning
        identity = torch.eye(
            temp_map2.shape[-1], device=temp_map2.device, dtype=temp_map2.dtype
        ).unsqueeze(0)
        temp_map1 = (temp_map1 + identity) / 2
        temp_map2 = (temp_map2 + identity) / 2

        # Compute the temporal effect by multiplying the regularized self-attention maps
        temporal_effect = torch.bmm(temp_map2, temp_map1)

        # Multiply temporal effect with cross_map to get accumulated relation
        # Adjust cross_map dimensions to match temporal_effect for matrix multiplication
        cross_map = cross_map.permute(1, 0, 2)  # [160, total spatial tokens, 77]
        accumulated_relation = torch.bmm(temporal_effect, cross_map)  # [160, 12, 77]

        # Permute and reshape accumulated_relation to match desired output format
        accumulated_relation = accumulated_relation.permute(1, 0, 2)  # [12, 160, 77]

        accumulated_relation = accumulated_relation.reshape(
            -1, self.attn_res[0], self.attn_res[1], accumulated_relation.shape[-1]
        )

        return accumulated_relation

    def preprocess_attention_flow_hard(self, attention_maps) -> torch.Tensor:

        cross_map, temp_map1, temp_map2 = attention_maps

        # Regularize the self-attention maps with identity matrix to stabilize learning
        identity = torch.eye(
            temp_map2.shape[-1], device=temp_map2.device, dtype=temp_map2.dtype
        ).unsqueeze(0)
        temp_map1 = (temp_map1 + identity) / 2
        temp_map2 = (temp_map2 + identity) / 2

        # Compute the temporal effect by multiplying the regularized self-attention maps
        temporal_effect = batch_min_max_matrix_multiplication(temp_map2, temp_map1)

        # Multiply temporal effect with cross_map to get accumulated relation
        # Adjust cross_map dimensions to match temporal_effect for matrix multiplication
        cross_map = cross_map.permute(1, 0, 2)  # [160, total spatial tokens, 77]
        accumulated_relation = batch_min_max_matrix_multiplication(
            temporal_effect, cross_map
        )  # [160, 12, 77]

        # Permute and reshape accumulated_relation to match desired output format
        accumulated_relation = accumulated_relation.permute(1, 0, 2)  # [12, 160, 77]

        accumulated_relation = accumulated_relation.reshape(
            -1, self.attn_res[0], self.attn_res[1], accumulated_relation.shape[-1]
        )

        return accumulated_relation

    def preprocess_attention_flow_soft(
        self, attention_maps, temperature=0.001
    ) -> torch.Tensor:

        cross_map, temp_map1, temp_map2 = attention_maps

        # Regularize the self-attention maps with identity matrix to stabilize learning
        identity = torch.eye(
            temp_map2.shape[-1], device=temp_map2.device, dtype=temp_map2.dtype
        ).unsqueeze(0)
        temp_map1 = (temp_map1 + identity) / 2
        temp_map2 = (temp_map2 + identity) / 2

        # Compute the temporal effect by multiplying the regularized self-attention maps
        temporal_effect = batch_soft_min_max_matrix_multiplication(
            temp_map2, temp_map1, temperature=temperature
        )

        # Multiply temporal effect with cross_map to get accumulated relation
        # Adjust cross_map dimensions to match temporal_effect for matrix multiplication
        cross_map = cross_map.permute(1, 0, 2)  # [160, total spatial tokens, 77]
        accumulated_relation = batch_soft_min_max_matrix_multiplication(
            temporal_effect, cross_map, temperature=temperature
        )  # [160, 12, 77]

        # Permute and reshape accumulated_relation to match desired output format
        accumulated_relation = accumulated_relation.permute(1, 0, 2)  # [12, 160, 77]

        accumulated_relation = accumulated_relation.reshape(
            -1, self.attn_res[0], self.attn_res[1], accumulated_relation.shape[-1]
        )

        return accumulated_relation

    def aggregate_attention_rollout(self, from_where: List[str]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        temp_per_spatial = 2
        attention_maps = self.get_average_attention()

        for location in from_where:
            for sid, (cross_map, num_heads) in enumerate(
                attention_maps[location]["spatial"]
            ):

                # Calculate the downsample rate based on the original and current attention map size
                cross_map = cross_map.reshape(
                    -1, num_heads, cross_map.shape[-2], cross_map.shape[-1]
                )
                cross_map = cross_map.permute(1, 0, 2, 3).mean(0)
                attn_chain = [cross_map]
                for tid in range(temp_per_spatial):

                    temp_map, num_heads = attention_maps[location]["temporal"][
                        sid * temp_per_spatial + tid
                    ]
                    temp_map = temp_map.reshape(
                        -1, num_heads, temp_map.shape[-2], temp_map.shape[-1]
                    )
                    temp_map = temp_map.permute(1, 0, 2, 3).mean(0)
                    attn_chain.append(temp_map)

                attn = self.preprocess_attention_rollout(attn_chain).unsqueeze(0)
                out.append(attn)

        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def aggregate_attention_flow_hard(self, from_where: List[str]) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        temp_per_spatial = 2
        attention_maps = self.get_average_attention()
        for location in from_where:
            for sid, (cross_map, num_heads) in enumerate(
                attention_maps[location]["spatial"]
            ):

                # Calculate the downsample rate based on the original and current attention map size
                cross_map = cross_map.reshape(
                    -1, num_heads, cross_map.shape[-2], cross_map.shape[-1]
                )
                cross_map = cross_map.permute(1, 0, 2, 3).mean(0)
                attn_chain = [cross_map]
                for tid in range(temp_per_spatial):

                    temp_map, num_heads = attention_maps[location]["temporal"][
                        sid * temp_per_spatial + tid
                    ]
                    temp_map = temp_map.reshape(
                        -1, num_heads, temp_map.shape[-2], temp_map.shape[-1]
                    )
                    temp_map = temp_map.permute(1, 0, 2, 3).mean(0)
                    attn_chain.append(temp_map)

                attn = self.preprocess_attention_flow_hard(attn_chain).unsqueeze(0)
                out.append(attn)

        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def aggregate_attention_flow_soft(
        self, from_where: List[str], temperature=0.001
    ) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        temp_per_spatial = 2
        attention_maps = self.get_average_attention()
        for location in from_where:
            for sid, (cross_map, num_heads) in enumerate(
                attention_maps[location]["spatial"]
            ):

                # Calculate the downsample rate based on the original and current attention map size
                cross_map = cross_map.reshape(
                    -1, num_heads, cross_map.shape[-2], cross_map.shape[-1]
                )
                cross_map = cross_map.permute(1, 0, 2, 3).mean(0)
                attn_chain = [cross_map]
                for tid in range(temp_per_spatial):

                    temp_map, num_heads = attention_maps[location]["temporal"][
                        sid * temp_per_spatial + tid
                    ]
                    temp_map = temp_map.reshape(
                        -1, num_heads, temp_map.shape[-2], temp_map.shape[-1]
                    )
                    temp_map = temp_map.permute(1, 0, 2, 3).mean(0)
                    attn_chain.append(temp_map)

                attn = self.preprocess_attention_flow_soft(
                    attn_chain, temperature=temperature
                ).unsqueeze(0)
                out.append(attn)

        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def __init__(self, attn_res, temporal_res):
        super().__init__(attn_res)
        self.temporal_res = temporal_res
