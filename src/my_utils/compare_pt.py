import torch

ckpt_a = torch.load("examples/scene_results/Mip-Nerf-360/bicycle/test_fraction_0.9/init_sfm/common_ckpt_5k/train_and_retrain/ckpts/ckpt_9999_rank0.pt", map_location="cpu")
ckpt_b = torch.load("examples/scene_results/Mip-Nerf-360/bicycle/test_fraction_0.9/init_sfm/common_ckpt_5k/novel_ratios_loss_1_sample_-1/refine_and_retrain/ckpts/ckpt_9999_rank0.pt", map_location="cpu")

sd_a = ckpt_a["state_dict"] if "state_dict" in ckpt_a else ckpt_a
sd_b = ckpt_b["state_dict"] if "state_dict" in ckpt_b else ckpt_b

assert(sd_a.keys() == sd_b.keys())

for k in sd_a["splats"]:
    print(k, sd_a["splats"][k].shape, sd_b["splats"][k].shape)

for k in sd_a["splats"]:
    a = sd_a["splats"][k]
    b = sd_b["splats"][k]

    if not torch.equal(a, b):
        print("DIFF:", k)