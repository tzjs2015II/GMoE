# GraphLoRA Implementation

Here is the code of **GraphLoRA**. It is based on the open-source repository "[mLoRA-0.3.2](https://github.com/mikecovlee/mLoRA/tree/0.3.2)". In addition to GraphLoRA, we have implemented the following models:

- **GraphLoRA**
- **MixLoRA** (already implemented in the mlora repository)
- **MoLA**
- **MING-MoE**
- **LoRAMoE**

We provide the training configuration files for all of the models mentioned above.

## Quick Start: Training Models

To start training a model, use the following command:

```bash
python mlora.py --base_model your_hf_style_base_model_dir --config your_path_to_the_config.json --seed 42 --log_file graphlora.log --bf16 --overwrite
```
To evaluate a model, use the following command:

```bash
python mlora.py --base_model your_hf_style_base_model_dir --config your_path_to_the_config.json --seed 42 --log_file graphlora.log --bf16 --overwrite --evaluate
```
Our model is trained on 1xA800(80G). If your memory is limited, you can use torch.checkpoint by deleting the "#" in mlora/trainer line 340, to save the memory
