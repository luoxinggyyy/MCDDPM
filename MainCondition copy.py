from DiffusionFreeGuidence.TrainCondition import train, eval


def main(model_config=None):
    modelConfig = {
        "state": "eval", # or eval or train
        "epoch": 1000,
        "batch_size": 12,
        "T": 1000,
        "channel": 32,
        "channel_mult": [1, 2, 2, 4],
        "num_res_blocks": 1,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 1.5,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 64,
        "grad_clip": 1.,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "F:/q/CheckpointsCondition/",
        "training_load_weight": None,
        "test_load_weight": "real_our_810_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow":12, 
        "sampledImgName1":"S.png",
        "label":0.2,
        "labelA":10,
        "labelB":5,
        "path":"/root/ddpm/data1/"

    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
