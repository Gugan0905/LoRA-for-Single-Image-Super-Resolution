# CS7180 - Advanced Perception - Final Project
## LoRA for Single Image Super Resolution


This repository houses the project work titled **Low Rank Adaptation in Deep Networks for Image Super Resolution** for the course "CS7180 Advanced Perception."

### Project Authors
- Gugan Kathiresan, NUID - 002756523, kathiresan.g@northeastern.edu
- Anirudh Muthuswamy, NUID - 002783250, muthuswamy.a@northeastern.edu
- Aditya Varshney, NUID - 002724777, varshney.ad@northeastern.edu

### Project Abstract
The pursuit of Single Image Super Resolution (SISR) in computer vision research remains steadfast.
However, newer, complex models like transformers, GANs, and diffusion networks often struggle with generating realistic results in unfamiliar domains.
Fine-tuning or training separate models for each new dataset proves computationally expensive and may not ensure satisfactory outcomes without extensive training.  
In response, the Low Rank Adaptation (LoRA) method, initially used for large language models, has garnered attention for its potential to expedite model adaptation.
This study aims to apply LoRA to prominent deep networks targeting single image super resolution, assessing its effectiveness in adapting to different data domains.

---

## Details

### Repository Highlights
- **Project Report:** [cs7180-AnirudhGuganAditya-FinalProject.pdf](cs7180-AnirudhGuganAditya-FinalProject.pdf)
- **Presentation:** [CS7180-FinalProject-Presentation.pdf](CS7180-FinalProject-Presentation.pdf)
- **Presentation Video:** [Google Drive Link](https://drive.google.com/file/d/1-Q1rFYcKoLLUWeFGq7emhFF3-wyjemnw/view?usp=sharing)
- **Demo Notebook:** [training_exp.ipynb](src/training_exp.ipynb)
- **Source Code:** Check the files and folders inside ```./src/```

### Operating System
- Google Colab Pro Ubuntu / T4 GPU
- Discovery Cluster Jupyter Notebook / V100 GPU
- macOS for file management

### Required Packages
In your command line:  
```cd /path/to/your/project```  
```pip install -r requirements.txt```  

### Compilation Instructions for Script Files
- Download Datasets:
  - [FloodNet dataset](https://drive.google.com/drive/folders/1sZZMJkbqJNbHgebKvHzcXYZHJd6ss4tH?usp=sharing)
  - [Set5 dataset](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)
  - [Set14 dataset](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)
- Unzip Files
- Place them in the working directory.
- Install Requirements
  - Use ```pip install -r requirements.txt.```
- Edit Parameters
  - Modify parameters in ```'src/main.py'``` as needed.
- Run Script 
  - Execute the ```main.py``` file using:  
  - ```!python ./src/main.py```

Feel free to explore the project report or any other content provided. If you have questions or need further assistance, don't hesitate to reach out!

## Project Findings

- Diffusion Models and Challenges
  - Exploration of diffusion models for image-to-image generative tasks revealed challenges in image generation beyond the source domain, affecting the quality of super resolution. Challenges and experiments with the Latent Diffusion Upscaler Model (LDM Upscaler) are discussed.
- Transformer Models and Challenges
  - The study employed variations of the Swin-Transformer for SISR. Challenges in training for super resolution and limitations in reconstruction and complexity are highlighted.
- GAN Models and Challenges
  - Employment of Enhanced Super Resolution GAN (ESRGAN) variants faced challenges in noise control, deep feature extraction, and upsampling for reconstruction.
- Applying LoRA
  - The concept and application of LoRA for model fine-tuning and reduction of trainable parameters are detailed, showcasing its effectiveness in improving super resolution quality and inference time.
- Experiments and Results
  - The study evaluates LoRA's effect on ESRGAN through pretraining, finetuning, and performance assessment across datasets (FloodNet, Set5, Set14). Results showcase improved PSNR, SSIM, and reduced parameters after LoRA application.

