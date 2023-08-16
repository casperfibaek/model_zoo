import sys; sys.path.append("../")

import torch
import torch.nn as nn
import torchmetrics

from utils import load_data, training_loop, TiledMSE


def train(
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    model: nn.Module = None,
    warmup_epochs: int = 10,
    learning_rate_end: float = 0.00001,
    name=str,
    use_wandb: bool = True,
    predict_func=None,
    save_best_model: bool = False,
    patience=20,
    min_epochs=50,
) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl_train, dl_val, dl_test = load_data(
        x="s2",
        y="area",
        with_augmentations=True,
        num_workers=0,
        batch_size=batch_size,
        encoder_only=False,
    )

    wmape = torchmetrics.WeightedMeanAbsolutePercentageError(); wmape.__name__ = "wmape"
    mae = torchmetrics.MeanAbsoluteError(); mae.__name__ = "mae"
    mse = torchmetrics.MeanSquaredError(); mse.__name__ = "mse"

    training_loop(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        learning_rate_end=learning_rate_end,
        model=model,
        # criterion=TiledMSE(0.2),
        criterion=nn.MSELoss(),
        device=device,
        metrics=[
            mse.to(device),
            wmape.to(device),
            mae.to(device),
        ],
        train_loader=dl_train,
        val_loader=dl_val,
        test_loader=dl_test,
        name=name,
        warmup_epochs=warmup_epochs,
        use_wandb=use_wandb,
        predict_func=predict_func,
        save_best_model=save_best_model,
        patience=patience,
        min_epochs=min_epochs,
    )

def predict_func(model, epoch):
    model.eval()
    model.to("cuda")
    
    img_path = "../data/images/naestved_s2.tif"
    img_arr = beo.raster_to_array(img_path, filled=True, fill_value=0, cast=np.float32) / 10000.0

    def predict(arr):
        swap = beo.channel_last_to_first(arr)
        as_torch = torch.from_numpy(swap).float()
        on_device = as_torch.to('cuda')
        predicted = model(on_device)
        on_cpu = predicted.cpu()
        as_numpy = on_cpu.detach().numpy()
        swap_back = beo.channel_first_to_last(as_numpy)

        return swap_back

    with torch.no_grad():
        predicted = beo.predict_array(
            img_arr,
            callback=predict,
            tile_size=64,
            n_offsets=3,
            batch_size=32,
        )
    beo.array_to_raster(
        predicted,
        reference=img_path,
        out_path=F"../visualisations/pred_{NAME}_{epoch}.tif",
    )


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=UserWarning)
    import buteo as beo
    import numpy as np

    NUM_EPOCHS = 100
    MIN_EPOCHS = 50
    WARMUP_EPOCHS = 10
    PATIENCE = 20
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    NAME = "MetaFormer05"

    depths = [3, 3, 3, 3]
    dims = [32, 48, 64, 80]

    # from model_Diamond import DiamondNet
    # model = DiamondNet(
    #     input_dim=10,
    #     output_dim=1,
    #     input_size=64,
    #     depths=depths,
    #     dims=dims,
    #     clamp_output=True,
    #     clamp_min=0.0,
    #     clamp_max=100.0,
    # )

    # from model_ResNet import ResNet
    # model = ResNet(
    #     input_dim=10,
    #     output_dim=1,
    #     depths=depths,
    #     dims=dims,
    #     clamp_output=True,
    #     clamp_min=0.0,
    #     clamp_max=100.0,
    # )

    # from model_CoreCNN import CoreUnet
    # model = CoreUnet(
    #     input_dim=10,
    #     output_dim=1,
    #     clamp_output=True,
    #     depths=depths,
    #     dims=dims,
    #     clamp_min=0.0,
    #     clamp_max=100.0,
    #     activation="relu",
    # )

    # from model_SENet import SENet
    # model = SENet(
    #     input_dim=10,
    #     output_dim=1,
    #     clamp_output=True,
    #     depths=depths,
    #     dims=dims,
    #     clamp_min=0.0,
    #     clamp_max=100.0,
    #     activation="relu",
    # )

    # from model_ConvNextV2 import ConvNextV2
    # model = ConvNextV2(
    #     input_dim=10,
    #     output_dim=1,
    #     clamp_output=True,
    #     depths=depths,
    #     dims=dims,
    #     clamp_min=0.0,
    #     clamp_max=100.0,
    # )

    # from model_vit import ViT
    # model = ViT(
    #     bchw=(BATCH_SIZE, 10, 64, 64),
    #     output_dim=1,
    #     patch_size=4,
    #     embed_dim=1024,
    #     n_layers=5,
    #     n_heads=16,
    # )

    # from model_vit import ViT
    # model = ViT(
    #     chw=(10, 64, 64),
    #     output_dim=1,
    #     patch_size=8,
    #     embed_dim=768,
    #     depth=3,
    #     num_heads=16,
    # )

    # from model_MixerMLP import MLPMixer
    # model = MLPMixer(
    #     chw=(10, 64, 64),
    #     output_dim=1,
    #     patch_size=8,
    #     embed_dim=512,
    #     dim=256,
    #     depth=5,
    # )

    # from model_DiamondFormer import DiamondFormer
    # model = DiamondFormer(
    #     chw=(10, 64, 64),
    #     output_dim=1,
    #     patch_size=8,
    #     # embed_dim=1024,
    #     # clamp_output=True,
    #     # clamp_min=0.0,
    #     # clamp_max=100.0,
    # )

    from model_MetaFormer import MetaFormer
    model = MetaFormer(
        chw=(10, 64, 64),
        output_dim=1,
        patch_size=8,
        depth=5,
        embed_dim=512,
        embed_channels=64,
        clamp_output=True,
        clamp_min=0.0,
        clamp_max=100.0,
    )

    train(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        min_epochs=MIN_EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        model=model,
        name=NAME,
        use_wandb=False,
        patience=PATIENCE,
        predict_func=predict_func,
    )

# DiamondNet - Default weights, batchNorm, ReLU
# Epoch 50/100: 100% 265/265 [01:51<00:00,  2.38it/s, loss=95.3480, mse=95.3480, wmape=0.6455, mae=3.0573, val_loss=27.0051, val_mse=27.0051, val_wmape=0.7505, val_mae=0.9521]
# Test Accuracy: 69.2184

# Best DiamondFormer
# Test Accuracy: 85.8081