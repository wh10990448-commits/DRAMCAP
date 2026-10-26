# DRAMCap
# DRAMCap: Dual-Directional ROI-Enhanced Image Captioning with Name-Attribute Dual-Memory Fusion

**Lulu Wang, Hao Wang, Ruoyu Zhang, Tongling Pan, Hai Sun, and Yingna Li**

## Abstract

## Dual-Directional ROI-Enhanced Image Captioning with Name-Attribute Dual-Memory Fusion
Image captioning aims to generate fluent descriptions. Retrieval-augmented methods prompt a frozen LLM with external memory, but a single global embedding loses object details and causes misbinding, while the memory indexes only names, ignoring attributes. We propose DRAMCap, a dual-directional Region of Interest (ROI)-enhanced captioner. It has three core components: (1) region-level dual-memory query—pooled tokens vote for names openly, and each region retrieves attributes from its predicted class subset, eliminating misbinding; (2) a shared lightweight Q-Former for refinement and fusion—it processes retrieved names and attributes and concatenates them with the final visual features, encoding into marker-free continuous prompts for frozen Vicuna-7B; (3) global-ROI dual feedback flow—a novel bidirectional mechanism that mutually enhances global and regional features for refined representations. Our model, with only a 7B LLM, surpasses 13B image captioning models on evaluation metrics and performs well on Flickr30k, NoCaps, and WHOOPS. The code is available [here](https://github.com/wh10990448-commits/DRAMCAP).

![DRAMCap architecture](assets/dramcap-framework.png)

*DRAMCap framework. Full-image and region-wise visual features interact with external name and attribute memories to condition the frozen language model. Flame and snowflake icons indicate trainable and frozen modules, respectively.*

**Lulu Wang, Hao Wang, Ruoyu Zhang, Tongling Pan, Hai Sun, and Yingna Li**
## Overview


![DRAMCap architecture from Figure 2 of the paper](assets/dramcap-framework.png)

*Figure 2 from the accompanying paper. The framework combines full-image and region-wise visual features, external name and attribute memories, and retrieval-and-fusion modules to condition the frozen language model. Flame and snowflake icons indicate trainable and frozen modules, respectively.*

The paper uses a frozen EVA-CLIP visual encoder, a frozen Vicuna-7B-v1.5 language decoder, and a trainable retrieval-and-fusion pipeline. Region features are obtained through 7 x 7 ROI-Align and per-ROI Q-Former processing. The name memory follows the LVIS-derived memory used by EVCap; the attribute memory is constructed from Visual Genome, VAW, and PACO.
├── assets/
│   ├── dramcap-framework.png      # Architecture figure from the paper
│   ├── dramcap-framework.png      # High-resolution architecture figure
│   └── dramcap-examples.png       # Qualitative examples from the paper

We acknowledge LAVIS, BLIP-2, EVCap, and the COCO, Flickr30k, LVIS, PACO, VAW, Visual Genome, NoCaps, and WHOOPS datasets and benchmarks. The architecture and qualitative-example images in this README are extracted from the accompanying DRAMCap manuscript.
We acknowledge LAVIS, BLIP-2, EVCap, and the COCO, Flickr30k, LVIS, PACO, VAW, Visual Genome, NoCaps, and WHOOPS datasets and benchmarks. The figures illustrate the framework and qualitative results presented in the accompanying DRAMCap manuscript.
