# MSWA: Refining Local Attention with Multi-ScaleWindow Attention

**Yixing Xu, Shivank Nag, Dong Li, Lu Tian, Emad Barsoum** | [Paper](https://arxiv.org/abs/2501.01039)

Advanced Micro Devices, Inc.

---

## Dependancies

```bash
torch == 2.1.2+rocm5.5
numpy == 1.24.4
einops == 0.7.0
peft == 0.10.0
datasets == 2.19.1
deepspeed == 0.14.1
wandb == 0.16.5
transformers == 4.34.0
accelerate == 0.29.2
tokenizers == 0.14.1
```

## Training

1. Download Redpajama dataset.

2. Prepare data. 

   ```bash
   python data_prepare.py
   ```

3. Run training script.

   ```bash
   sh script/diff_run.sh
   ```

## Citation

```
@article{xu2025mswa,
  title={MSWA: Refining Local Attention with Multi-ScaleWindow Attention},
  author={Xu, Yixing and Nag, Shivank and Li, Dong and Tian, Lu and Barsoum, Emad},
  journal={arXiv preprint arXiv:2501.01039},
  year={2025}
}
```

