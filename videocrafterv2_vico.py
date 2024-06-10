import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import numpy as np
import argparse
import random
from vico.video_crafter_diffusers.unet_3d_videocrafter import (
    UNet3DVideoCrafterConditionModel,
)
from vico.video_crafter_diffusers.pipeline_text_to_video_videocrafter import (
    export_to_video,
)
from vico.video_crafter_diffusers.pipeline_text_to_video_videocrafter_vico import (
    TextToVideoVideoCrafterAttributetoComposePipeline,
)

from utils import find_keywords

# load args with argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompts",
    type=str,
    default="A Retriever is driving a Porsche car.",
)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--H", type=int, default=320)
parser.add_argument("--W", type=int, default=512)
parser.add_argument("--token_indices", type=str, default="2,4,7")
parser.add_argument("--guidance_scale", type=float, default=12.0)
parser.add_argument("--num_frames", type=int, default=16)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--max_iterations", type=int, default=10)
parser.add_argument("--init_step_size", type=float, default=0.0001)
parser.add_argument("--attribute_mode", type=str, default="latent_attention_flow_st_soft")
parser.add_argument(
    "--unet_path",
    type=str,
    default="PATH-TO/videocrafterv2_diffusers",
)
parser.add_argument("--output_video_path", type=str, default="./video_576_car_dog.mp4")
args = parser.parse_args()

generator = torch.Generator("cuda").manual_seed(args.seed)
random.seed(args.seed)

pipe = TextToVideoVideoCrafterAttributetoComposePipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w", torch_dtype=torch.float16
)
pipe.unet = UNet3DVideoCrafterConditionModel.from_pretrained(
    args.unet_path,
    torch_dtype=torch.float16,
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config, use_karras=True, algorithm_type="sde-dpmsolver++"
)
pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")

attn_res = (int(args.H / 32), int(args.W / 32))
print(attn_res)

# token_indices = [i+1 for i, token in enumerate(args.prompts.split(' ')) if token in ['Retriever', 'car']]
# token_indices = [2, 4, 7]
if args.token_indices == "":

    token_indices = find_keywords(args.prompts)
    token_indices = [i + 1 for i in token_indices]
else:
    token_indices = [int(i) for i in args.token_indices.split(",")]
print(pipe.get_indices(args.prompts))
print("token_indices", token_indices)

video_frames = pipe(
    args.prompts,
    token_indices=token_indices,  # [0, 1] is the index of the attributes ["car", "dog"]
    num_inference_steps=args.steps,
    guidance_scale=args.guidance_scale,
    height=args.H,
    width=args.W,
    num_frames=args.num_frames,
    fps=args.fps,
    generator=generator,
    attn_res=attn_res,
    init_step_size=args.init_step_size,
    max_iterations=args.max_iterations,
    attribute_mode=args.attribute_mode,
).frames

if video_frames.shape[0] == 1:
    video_frames = video_frames[0]
    video_path = export_to_video(
        video_frames, output_video_path=args.output_video_path, fps=8
    )

elif video_frames.shape[0] > 1:
    raw_video_frames = video_frames[0]
    video_path = export_to_video(
        raw_video_frames, output_video_path=args.output_video_path, fps=8
    )
    for i, idx in enumerate(token_indices):
        video_path = export_to_video(
            video_frames[i + 1],
            output_video_path=args.output_video_path.replace(".mp4", f"{idx}.mp4"),
            fps=8,
        )
