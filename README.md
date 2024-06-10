# Vico
This reposioty contains our official implementation for **Vico**. Vico provides a unified solution for compositional video generation by equalizing the information flow of text tokens.

**Compositional Video Generation as Flow Equalization**

🥯[[Project Page](https://adamdad.github.io/vico/)] 📝[[Paper](https://arxiv.org/abs/2404.06091)] </>[[code](https://github.com/Adamdad/vico)]

Xingyi Yang, Xinchao Wang

National University of Singapore



![pipeline](assets/pipeline.jpg)


> We introduce Vico, a generic framework for compositional video generation that explicitly ensures all concepts are represented properly. At its core, Vico analyzes how input tokens influence the generated video, and adjusts the model to prevent any single concept from dominating. We apply our method to multiple diffusion-based video models for compositional T2V and video editing. Empirical results demonstrate that our framework significantly enhances the compositional richness and accuracy of the generated videos.

# Installation
- Enviroments
    ```shell
    pip install diffusers==0.26.3
    ```

- For VideoCrafterv2, it is recommanded to download the `diffusers` checkpoints first on (`adamdad/videocrafterv2_diffusers`)[https://huggingface.co/adamdad/videocrafterv2_diffusers]
    ```shell
    git lfs install
    git clone https://huggingface.co/adamdad/videocrafterv2_diffusers
    ```


# Usage
```shell
export PYTHONPATH="$PWD"
python videocrafterv2_vico.py --prompts XXX --attribution_mode "latent_attention_flow_st_soft" 
```