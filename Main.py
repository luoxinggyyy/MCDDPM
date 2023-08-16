from Diffusion.Train import train, eval


def main(model_config = None):
    modelConfig = {
        "state": "eval", # or eval
        "epoch": 500,
        "batch_size": 8,
        "T": 1000,
        "channel": 16,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 1,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 64,
        "grad_clip": 1.,
        "device": "cuda",
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_499_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
