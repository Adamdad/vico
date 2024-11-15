from .attention_attribution import (
    AttentionAttribution,
    AttentionIoUAttribution,
    AttentionVisualizer,
    AttentionSTRollOutAttribution,
    AttentionRollOutVisualizer,
    AttentionSTFlowAttribution,
    AttentionFlowVisualizer
)


def get_attributer(attribution_mode, unet, device, attn_res, image_size=None, temporal_res=12):
    if attribution_mode == "latent_attention":
        return AttentionAttribution(unet, device, attn_res)
    elif attribution_mode == "latent_attentioniou":
        return AttentionIoUAttribution(unet, device, attn_res)
    elif attribution_mode in ["latent_attention_flow_st_hard"]:
        print("............ Using AttentionSTFlowAttribution Hard............")
        return AttentionSTFlowAttribution(unet, device, attn_res, temporal_res, mode = 'flow_hard')
    elif attribution_mode in ["latent_attention_flow_st_soft"]:
        print("............ Using AttentionSTFlowAttribution Soft............")
        return AttentionSTFlowAttribution(unet, device, attn_res, temporal_res, mode = 'flow_soft')
    elif attribution_mode in ["cogvideox_attention_flow_st_soft"]:
        print("............ Using AttentionSTFlowAttribution Soft............")
        return AttentionSTFlowAttribution(unet, device, attn_res, temporal_res, mode = 'flow_soft')
    elif attribution_mode == "attention_vis":
        print("............ Using AttentionVisualizer ............")
        return AttentionVisualizer(unet, device, image_size=image_size)
    elif attribution_mode in ["attention_flow_vis_hard"]:
        print("............ Using AttentionFlowVisualizer Hard ............")
        return AttentionFlowVisualizer(unet, device, image_size=image_size, mode='flow_hard')
    elif attribution_mode in ["attention_flow_vis_soft"]:
        print("............ Using AttentionFlowVisualizer Soft ............")
        return AttentionFlowVisualizer(unet, device, image_size=image_size, mode='flow_soft')
    else:
        raise ValueError(f"attribution_mode: {attribution_mode} is not supported")
