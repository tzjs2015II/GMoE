# GMoE: Empowering LLMs Fine-Tuning via MoE Graph Collaboration

This repository contains the code for the paper “GMoE: Empowering LLMs Fine-Tuning via MoE Graph Collaboration”. The project is based on the open-source repository "[mLoRA-0.3.2](https://github.com/mikecovlee/mLoRA/tree/0.3.2)". GMoE is a new MoE architecture with a Graph-based router, Poisson distribution-based distinction strategy and Normal distribution-based balance strategy.

## File description

- **config**: Including the configurations of training or evaluating

- **gmoe/backends**: Some backend tools for GMoE.
- **gmoe/common**: The implementation of Transformer architecture.
- **gmoe/models**: The implementation of some series of Transformer-based models.
- **gmoe/tasks**: The implementation of datasets.
- **GMoE.py** The start file of this project.

## Environment Requirements
- python=3.11, pytorch>=2.1.2, pyg
- Other dependencies,, See ```bash requirements.txt```

## Quick Start
### STEP 1: Download base models

- [Qwen2 7B](https://huggingface.co/Qwen/Qwen2-7B)
  
- [Llama3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
  
- [Yi 9B](https://huggingface.co/01-ai/Yi-9B)
### STEP 2: Prepare the configs

Configure the configs at folder```bash config```. We have already given the config of **GMoE** and other four baseline models: **LoRAMoE**, **MingMoE**, **MoLA** and **MixLoRA**.

### STEP 3: Start training

Replace the **[base model]** and the **[train/evaluate config]** below with the directory of base model and the configuration in Folder "config".

``````python 
python GMoE.py --base_model [base model] --config [train config] --seed 42 --log_file GMoE.log --bf16 --overwrite
``````



### STEP 4: Conduct evaluation

After training process, we can conduct the evaluation step with the command below:

``````python
python GMoE.py --base_model [base model] --config [train config] --seed 42 --log_file GMoE.log --bf16 --overwrite --evaluate
``````

***Note***:   **Do not** change the information in the **train config** after training step, or it won't find the right adapter.

## Citation

If you use this project in your research, please cite the following paper:

```
@misc{bai2025gmoeempoweringllmsfinetuning,
      title={GMoE: Empowering LLMs Fine-Tuning via MoE Graph Collaboration}, 
      author={Ting Bai and Yue Yu and Le Huang and Zenan Xu and Zhe Zhao and Chuan Shi},
      year={2025},
      eprint={2412.16216},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.16216}, 
}
```



