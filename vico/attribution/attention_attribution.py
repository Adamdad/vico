import torch
import numpy as np
from typing import List

import abc
from torch.nn import functional as F
import math
from .attention_processor import (
    AttendExciteAttnProcessor,
    AttendExciteAttnProcessorSpatialTemporal,
    AttendVisProcessor,
    AttendVisProcessorSpatialTemporal,
)
from .attention_store import (
    AttentionStore,
    AttentionSTStore,
    AttentionVisStore,
    AttentionRolloutVisStore,
    AttentionFlowVisStore
)
import cv2
import PIL
from torch import optim


class BaseAttentionAttribution:
    """
    Base class for attention attribution in a neural network model.

    Args:
        unet (object): The neural network model.
        device (str): The device to run the model on.
        attn_res (int): The resolution of the attention map.

    Attributes:
        unet (object): The neural network model.
        device (str): The device to run the model on.
        attention_store (object): The attention store object.
        step_size (float): The step size for updating the latent.
    """

    def __init__(self, unet, device, attn_res):
        self.unet = unet
        self.device = device
        self.attn_res = attn_res
        self.set_attention_store()
        self.register_attention_control()

    def set_attention_store(self):
        self.attention_store = AttentionStore(self.attn_res)

    def set_optimizer(self, lr, opt="sgd"):
        """
        Set the step size for updating the latent.

        Args:
            step_size (float): The step size for updating the latent.
        """
        self.lr = lr
        self.opt = opt

    def register_attention_control(self):
        """
        Register the attention control for the neural network model.
        """
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if "temp" in name or "transformer_in" in name:
                attn_procs[name] = self.unet.attn_processors[name]
                continue

            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(
                attnstore=self.attention_store, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    def _update_latent(self, latents, loss, selected_index) -> torch.Tensor:
        """
        Update the latent representation using the computed loss. using the Adam optimizer.
        """
        # Clear gradients before backward pass
        # Ensure loss is set to require gradients
        # Compute gradients w.r.t. latents
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        # print(self.latents)
        self.latents.grad = grad_cond
        # grad clipping
        # torch.nn.utils.clip_grad_norm_(self.latents, 1.0)
        
        self.optimizer.step()
        
        # Clear gradients after updating
        self.optimizer.zero_grad()
        # print(self.latents)
        
        return self.latents

    def _update_textembed(self, text_weight, loss, selected_index) -> torch.Tensor:
        """
        Update the text embedding using the computed loss. using the Adam optimizer.
        """
        # Clear gradients before backward pass
        # Ensure loss is set to require gradients
        # Compute gradients w.r.t. latents
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [text_weight], retain_graph=True
        )[0]
        # print(self.latents)
        self.text_weight.grad = grad_cond
        # grad clipping
        # torch.nn.utils.clip_grad_norm_(self.latents, 1.0)
        
        self.optimizer.step()
        
        # Clear gradients after updating
        self.optimizer.zero_grad()
        # print(self.latents)
        
        return self.text_weight
    
    # def _update_latent_sgd(self, latents, loss, selected_index) -> torch.Tensor:
    #     """
    #     Update the latent representation using the computed loss.
    #     """
    #     # Compute gradients w.r.t. latents
    #     grad_cond = torch.autograd.grad(
    #         loss.requires_grad_(True), [latents], retain_graph=True
    #     )[0]

    #     # Update the latent at the selected index
    #     latents = latents - self.lr * grad_cond

    #     return latents

    # def _update_latent_momentum(self, latents, loss, selected_index) -> torch.Tensor:
    #     """
    #     Update the latent representation using the computed loss.
    #     """
    #     # Compute gradients w.r.t. latents
    #     grad_cond = torch.autograd.grad(
    #         loss.requires_grad_(True), [latents], retain_graph=True
    #     )[0]

    #     # Update the latent at the selected index
    #     self.v[:, :, selected_index] = (
    #         self.beta1 * self.v[:, :, selected_index] - self.lr * grad_cond
    #     )
    #     latents = latents + self.v[:, :, selected_index]

    #     return latents

    # def _update_latent(self, latents, loss, selected_index) -> torch.Tensor:
    #     if self.opt == "adam":
    #         return self._update_latent_adam(latents, loss, selected_index)
    #     elif self.opt == "sgd":
    #         return self._update_latent_sgd(latents, loss, selected_index)
    #     elif self.opt == "momentum":
    #         return self._update_latent_momentum(latents, loss, selected_index)
    #     else:
    #         raise ValueError(f"Invalid optimizer: {self.opt}")

    @abc.abstractmethod
    def update_latent(self):
        """
        Abstract method for updating the latent.
        """
        pass

    @abc.abstractmethod
    def update_textembed(self):
        """
        Abstract method for updating the text embedding.
        """
        pass

    @abc.abstractmethod
    def compute_loss(self):
        """
        Abstract method for computing the loss.
        """
        pass


class AttentionVisualizer(BaseAttentionAttribution):
    def __init__(self, unet, device, image_size):
        self.image_size = image_size
        self.attention_map = []

        super().__init__(unet, device, attn_res=None)

    def enable_vis(self, enable: bool):
        self.attention_store.enable_vis = enable

    def register_attention_control(self):
        """
        Register the attention control for the neural network model.
        """
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if "temp" in name or "transformer_in" in name:
                attn_procs[name] = self.unet.attn_processors[name]
                continue

            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendVisProcessor(
                attnstore=self.attention_store, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    def set_attention_store(self):
        self.attention_store = AttentionVisStore(self.attn_res, self.image_size)

    def compute_attention_maps(self, indices: List[int]):
        attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"),
        )
        self.attention_map.append(attention_maps)

        return attention_maps

    def add_attention_maps_to_video(self, video, indices, alpha=0.7, normalize=True, seg=False, thresh='mean'):

        print("Adding attention maps to video")
        print(f"Attention map length: {len(self.attention_map)}")
        print(f"Attention map shape: {self.attention_map[0].shape}")
        
        # heatmap shape is numpy array (temporal_res, H, W)
        # video shape is numpy array (temporal_res, H, W, 3)
        # Stack all attention maps and calculate the mean to create a single heatmap
        video = video[0]
        # if isinstance(video[0], PIL.Image.Image):
        #     video = [np.array(frame) for frame in video]
        #     video = np.array(video)
            
        attention_map_stack = np.stack(self.attention_map)
        all_mean_heatmap = attention_map_stack.sum(axis=0)
        all_videos = []
        for idx in indices:
            print(f"Adding attention map for index {idx}")
            temp_video = video.copy()
            mean_heatmap = all_mean_heatmap[:, :, :, idx].copy()
            
            if thresh == 'mean':
                thresh = np.mean(mean_heatmap)
            binary_heatmap = np.float32(mean_heatmap>thresh)
            
            # normalize the heatmap
            if normalize:
                mean_heatmap = (mean_heatmap - mean_heatmap.min()) / (
                    mean_heatmap.max() - mean_heatmap.min()
                )
            mean_heatmap = mean_heatmap * binary_heatmap

            # Blend the colored heatmap with each video frame
            for i in range(len(temp_video)):
                colored_heatmap = cv2.applyColorMap(
                    np.uint8(mean_heatmap[i] * 255), cv2.COLORMAP_JET
                )
                colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

                # convert to video type, 0-1
                colored_heatmap = colored_heatmap.astype(temp_video.dtype) / 255.0
                # Blend the heatmap with the video frame
                temp_video[i] = cv2.addWeighted(
                    temp_video[i], alpha, colored_heatmap, 1 - alpha, 0
                )

            # add a dim to first axis
            temp_video = np.expand_dims(temp_video, axis=0)
            all_videos.append(temp_video)
            
            
        if seg:
            # ========================================================================================================
            for idx in indices:
                print(f"Adding attention map for index {idx}")
                # temp_video = video.copy()
                mean_heatmap = all_mean_heatmap[:, :, :, idx].copy()
                print(mean_heatmap.min(), mean_heatmap.max())
                if thresh == 'mean':
                    thresh = np.mean(mean_heatmap)
                binary_heatmap = np.float32(mean_heatmap>thresh)
                # shape of 24, 320, 512 -> 24, 320, 512, 3
                binary_heatmap = np.stack([binary_heatmap]*3, axis=-1)

                # add a dim to first axis
                temp_video = np.expand_dims(binary_heatmap, axis=0)
                
                all_videos.append(temp_video)
            print([video.shape for video in all_videos])
            
        return np.concatenate(all_videos, axis=0)


class AttentionSTVisualizer(AttentionVisualizer):

    def register_attention_control(self):
        """
        Register the attention control for the neural network model.
        """
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            # print(name)
            if "transformer_in" in name:
                attn_procs[name] = self.unet.attn_processors[name]
                continue
            # for animatediff, we omit the last down_block and the first up_block, 
            # since no cross attention is there
            if 'down_blocks.3.motion_modules' in name or 'up_blocks.0.motion_modules' in name:
                attn_procs[name] = self.unet.attn_processors[name]
                continue

            if "temp" in name or 'motion_modules' in name:
                mode = "temporal"
            else:
                mode = "spatial"

            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendVisProcessorSpatialTemporal(
                attnstore=self.attention_store, place_in_unet=place_in_unet, mode=mode
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count


class AttentionRollOutVisualizer(AttentionSTVisualizer):

    def set_attention_store(self):
        self.attention_store = AttentionRolloutVisStore(self.attn_res, self.image_size)
        
    def set_text_attention(self, text_attentions):
        rollout = None
            
        for i, text_attention in enumerate(text_attentions[::-1]):
            text_attention = torch.mean(text_attention, dim=1)
            # Regularize the self-attention maps with identity matrix to stabilize learning
            identity = torch.eye(text_attention.shape[-1], device=text_attention.device, dtype=text_attention.dtype).unsqueeze(0)
            text_attention = (text_attention + identity) / 2
            
            if rollout is None:
                
                rollout = text_attention
            else:
                rollout = torch.bmm(rollout, text_attention)
                

        diag = rollout[0].diagonal()
        # normalize the rollout by diagonal values
        rollout = rollout / diag.unsqueeze(0)
            
        self.attention_store.text_attention = rollout


class AttentionFlowVisualizer(AttentionSTVisualizer):
    def __init__(self, unet, device, image_size, mode="flow_hard"):
        self.mode = mode
        super().__init__(unet, device, image_size)
        
    def set_attention_store(self):
        self.attention_store = AttentionFlowVisStore(self.attn_res, self.image_size, mode=self.mode)


class AttentionAttribution(BaseAttentionAttribution):
    """
    Class for computing attention attribution using the attend-and-excite method.

    Args:
        unet (object): The U-Net model.
        device (str): The device to run the computations on.
        attn_res (int): The resolution of the attention maps.
        step_size (float, optional): The step size for updating the latent representation. Defaults to 0.1.
    """

    def __init__(self, unet, device, attn_res):
        super().__init__(unet, device, attn_res)
        self.last_idx = -1

    def set_optimizer(
        self, latents=None, text_weight=None, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, opt="momentum"
    ):
        assert latents is not None or text_weight is not None, "Latents or text weight must be provided"
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.opt = opt
        self.latents = latents
        self.text_weight = text_weight
        param = []
        
        if self.latents is not None:
            param.append(self.latents)
        if self.text_weight is not None:
            param.append(self.text_weight)
            
            
        if opt == "adam":
            self.optimizer = optim.Adam(param, lr=lr, eps=epsilon)
        elif opt == "momentum":
            self.optimizer = optim.SGD(param, lr=lr, momentum=beta1)
        elif opt == "sgd":
            self.optimizer = optim.SGD(param, lr=lr)

    def _compute_loss(
        self, max_attention_per_index: List[torch.Tensor]
    ) -> torch.Tensor:
        """Computes the attend-and-excite loss using the maximum attention value for each token."""
        losses = [max(0, 1.0 - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        return loss

    def _compute_max_attention_per_index(
        self,
        attention_maps: torch.Tensor,
        indices: List[int],
    ) -> List[torch.Tensor]:
        """Computes the maximum attention value for each of the tokens we wish to alter."""

        attention_for_text = attention_maps[:, :, :, 1 : self.last_idx]

        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices = [index - 1 for index in indices]

        # Extract the maximum values
        max_indices_list = []
        for i in indices:
            image = attention_for_text[:, :, :, i]
            smoothing = GaussianSmoothing(dim=3).to(attention_maps.device)
            input = F.pad(
                image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode="reflect"
            )
            image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())

        return max_indices_list

    def update_latent(self, latent, loss, selected_index: List[int]):
        """
        Updates the latent representation based on the attention maps and specified indices.

        Args:
            latent (torch.Tensor): The latent representation to be updated.
            indices (List[int]): The indices of the tokens to be altered.

        Returns:
            torch.Tensor: The updated latent representation.
        """

        latent = self._update_latent(latent, loss, selected_index)
        self.attention_store.reset()
        return latent
    
    def update_text_weight(self, text_weight, loss, selected_index: List[int]):
        text_weight = self._update_textembed(text_weight, loss, selected_index)
        self.attention_store.reset()
        
        return text_weight

    def compute_loss(self, indices: List[int]):
        attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"),
        )
        max_attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices=indices,
        )
        loss = self._compute_loss(max_attention_per_index)
        return loss


class AttentionIoUAttribution(AttentionAttribution):

    def _compute_attention_per_index(
        self,
        attention_maps: torch.Tensor,
        indices: List[int],
    ) -> List[torch.Tensor]:
        """Computes the maximum attention value for each of the tokens we wish to alter."""

        attention_for_text = attention_maps[:, :, :, 1 : self.last_idx]

        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices = [index - 1 for index in indices]

        # Extract the maximum values
        indices_list = []
        for i in indices:
            image = attention_for_text[:, :, :, i]
            smoothing = GaussianSmoothing(dim=3).to(attention_maps.device)
            input = F.pad(
                image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode="reflect"
            )
            image = smoothing(input).squeeze(0).squeeze(0)
            indices_list.append(image)

        return indices_list

    def _compute_loss(self, attention_per_index: List[torch.Tensor]) -> torch.Tensor:
        # I want a loss, that (1) no interation between the attention maps of different tokens
        # (2) the smaller attention map should not be too small

        # Compute the intersection over union
        intersection = attention_per_index[0]

        for i in range(len(attention_per_index)):
            for j in range(i + 1, len(attention_per_index)):
                intersection = torch.min(intersection, attention_per_index[i])
                union = torch.max(attention_per_index[0], attention_per_index[i])
                iou = intersection.sum() / union.sum()
                loss = iou

        loss += (
            torch.max(
                [
                    1 - attention_per_index[i].max()
                    for i in range(len(attention_per_index))
                ]
            )
            * 0.5
        )
        return loss

    def compute_loss(self, indices: List[int]):
        attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"),
        )
        attention_per_index = self._compute_attention_per_index(
            attention_maps=attention_maps,
            indices=indices,
        )
        loss = self._compute_loss(attention_per_index)
        return loss


class AttentionSTAttribution(AttentionAttribution):
    def __init__(self, unet, device, attn_res, temporal_res=16):
        self.temporal_res = temporal_res
        super().__init__(unet, device, attn_res)

    def set_attention_store(self):
        self.attention_store = AttentionSTStore(self.attn_res, self.temporal_res)

    def register_attention_control(self):
        """
        Register the attention control for the neural network model.
        """
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            # print(name)
            if "transformer_in" in name:
                attn_procs[name] = self.unet.attn_processors[name]
                continue

            if "temp" in name or 'motion_modules' in name:
                mode = "temporal"
            else:
                mode = "spatial"

            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessorSpatialTemporal(
                attnstore=self.attention_store, place_in_unet=place_in_unet, mode=mode
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count


class AttentionSTRollOutAttribution(AttentionSTAttribution):
    
    def _compute_max_attention_per_index(
        self,
        attention_maps: torch.Tensor,
        indices: List[int],
    ) -> List[torch.Tensor]:
        """Computes the maximum attention value for each of the tokens we wish to alter."""

        attention_for_text = attention_maps[:, :, :, 1 : self.last_idx]

        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices = [index - 1 for index in indices]

        # Extract the maximum values
        max_indices_list = []
        for i in indices:
            image = attention_for_text[:, :, :, i]
            smoothing = GaussianSmoothing(dim=3).to(attention_maps.device)
            input = F.pad(
                image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode='reflect'
            )
            image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())

        return max_indices_list
    def compute_loss(self, indices: List[int]):
        attention_rollouts = self.attention_store.aggregate_attention_rollout(
            from_where=("up", "down", "mid"),
        )

        attention_rollouts = self._compute_max_attention_per_index(
            attention_maps=attention_rollouts,
            indices=indices,
        )
        loss = self._compute_loss(attention_rollouts)

        return loss

class AttentionSTFlowAttribution(AttentionSTAttribution):
    def __init__(self, unet, device, attn_res, temporal_res=16, mode="flow_hard"):
        self.mode = mode
        self.temperature = 0.01
        super().__init__(unet, device, attn_res, temporal_res)
        

    def compute_loss(self, indices: List[int]):
        if self.mode == "flow_hard":
            attention_flows = self.attention_store.aggregate_attention_flow_hard(
                from_where=("up", "down", "mid"),
            )
        elif self.mode == "flow_soft":
            attention_flows = self.attention_store.aggregate_attention_flow_soft(
                from_where=("up", "down", "mid"),
                temperature=self.temperature,
            )

        attention_flows = self._compute_max_attention_per_index(
            attention_maps=attention_flows,
            indices=indices,
        )
        loss = self._compute_loss(attention_flows)

        return loss



class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)
