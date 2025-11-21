# RESAnything: Attribute Prompting for Arbitrary Referring Segmentation
[![Project Page](https://img.shields.io/badge/RESAnything-Website-pink?logo=googlechrome&logoColor=pink)](https://suikei-wang.github.io/RESAnything/)
[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2505.02867)

<div>
  <a href="https://suikei-wang.github.io/">Ruiqi Wang</a>,
  <a href="https://www2.cs.sfu.ca/~haoz/">Hao Zhang</a>
  <br>
  Simon Fraser University
</div>

<div>
  <i>This repository contains a re-implementation of the codebase. The initial version is subject to a protected license that restricts redistribution. The prompts provided in this repository may not be the original version, and optimal performance may require further iterative refinement and tuning.</i>
</div>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="./teaser.png">
    Open-vocabulary and zero-shot referring expression segmentation with RESAnything.
</div>



## 🔧 Installation

### 1. Environment Setup

```bash
conda create -n resanything python=3.9
conda activate resanything
pip install -r requirements.txt
```


### 2. Download Required Models

#### Download SAM ViT-H Model
```bash
cd RESAnything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### Download Qwen2.5-VL-7B-Instruct (you may use other backbone models)
The model will be automatically downloaded from Hugging Face on first run, or you can pre-download:

```bash
# Using Hugging Face CLI
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./Qwen2.5-VL-7B-Instruct
```


## 🚀 Quick Start

### Single Image Demo

Process a single image with a referring expression:

**Example:**
```bash
python demo.py \
    --image_path "test.jpg" \
    --input_expression "left person" \
    --output_path "./output"
```

### Batch Processing

Process multiple images with different expressions:

1. **Create an expressions file** (`expressions.txt`) with format `imagename|expression`:
```
image1.jpg|the red car in the center
image2.jpg|the dog sitting on the grass  
image3.jpg|the building with blue windows
```

2. **Run batch processing:**
```bash
python demo_batch.py \
    --images_folder "path/to/images" \
    --expressions_file "expressions.txt" \
    --output_path "path/to/output"
```

## 🔬 Configuration

Edit `config.yaml` to customize model and processing parameters:

```yaml
model:
  name: "Qwen/Qwen2.5-VL-7B-Instruct"
  dtype: "bfloat16"
  attn_implementation: "flash_attention_2"
  device_map: "auto"

sam:
  checkpoint: "sam_vit_h_4b8939.pth"
  model_type: "vit_h"
  device: "cuda"
  min_mask_percentage: 0.004
  points_per_side: 16
  pred_iou_thresh: 0.92
  stability_score_thresh: 0.92

batch_size: 16
```
You may also edit `prompts.yaml` for better performance or domain-specific tasks. 


### Performance Tips

- Use `bfloat16` precision for better memory efficiency
- Enable `flash_attention_2` for faster processing
- Adjust SAM parameters for speed vs. quality trade-off

## 📁 File Structure

```
RESAnything/
├── config.py              # Configuration loader
├── config.yaml           # Main configuration file
├── demo.py               # Single image processing
├── demo_batch.py         # Batch processing
├── generation.py         # Text generation utilities
├── prompts.py           # Prompt management
├── prompts.yaml         # Prompt templates
├── sam_utils.py         # SAM processing utilities
├── similarity.py        # Similarity computation
├── requirements.txt     # Python dependencies
├── sam_vit_h_4b8939.pth # SAM model checkpoint
└── Qwen2.5-VL-7B-Instruct/ # Qwen model directory
```

## 📊 ABO-ARES Dataset
We will try to release ABO-ARES dataset ASAP. 

## 👨‍💻 Authors & Contact

If you have any queries, feel free to contact: Ruiqi Wang (rwa135@sfu.ca)

## 📚 Bibtex

```bibtex
@inproceedings{wang2025resanything,
  title={Resanything: Attribute prompting for arbitrary referring segmentation},
  author={Wang, Ruiqi and Zhang, Hao},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
