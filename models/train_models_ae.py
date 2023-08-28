import sys; sys.path.append("../")

import torch
import torch.nn as nn
import torchmetrics

from utils import training_loop, load_data_ae, TiledMAPE, TiledMSE, TiledMAPE2
import buteo as beo
import numpy as np


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

    dl_train, dl_val, dl_test = load_data_ae(
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
        # criterion=nn.MSELoss(),
        # criterion=TiledMSE(bias=0.8),
        # criterion=TiledMAPE(beta=0.1, bias=0.8),
        criterion=TiledMAPE2(beta=0.1, bias=0.8),
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

img_path = "../data/images/naestved_s2.tif"
img_arr = beo.raster_to_array(
    img_path,
    filled=True,
    fill_value=0,
    cast=np.float32,
    pixel_offsets=[1000, 500, 64, 64],
) / 10000.0


def predict_func(model, epoch):
    model.eval()
    model.to("cuda")

    with torch.no_grad():
        swap = beo.channel_last_to_first(img_arr)[np.newaxis, ...]
        as_torch = torch.from_numpy(swap).float()
        on_device = as_torch.to('cuda')
        predicted = model(on_device)
        on_cpu = predicted.cpu()
        as_numpy = on_cpu.detach().numpy()
        pred = beo.channel_first_to_last(as_numpy)[0, ...]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    q01 = np.quantile(img_arr, 0.01)
    q99 = np.quantile(img_arr, 0.99)
    ax1.imshow(img_arr[:, :, 2], vmin=q01, vmax=q99, cmap="gray")
    ax1.set_title('Image 1')
    ax1.axis('off')

    ax2.imshow(pred[:, :, 2], vmin=q01, vmax=q99, cmap="gray")
    ax2.set_title('Image 2')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(F"../visualisations/pred_{NAME}_{epoch}.png")
    plt.close()


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=UserWarning)
    import buteo as beo
    import numpy as np
    import matplotlib.pyplot as plt

    NUM_EPOCHS = 150
    MIN_EPOCHS = 50
    WARMUP_EPOCHS = 10
    PATIENCE = 20
    LEARNING_RATE = 0.001
    LEARNING_RATE_END = 0.00001
    BATCH_SIZE = 16
    NAME = "MAE01"

    depths = [3, 3, 3, 3]
    dims = [32, 48, 64, 80]

    from model_MixerMLP import MLPMixer
    model = MLPMixer(
        chw=(10, 64, 64),
        output_dim=10,
        patch_size=4,
        embed_dim=1024,
        expansion=4,
        drop_n=0.1,
        drop_p=0.0,
        dim=512,
        depth=5,
    )
    weights_path = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/model_zoo/trained_models/MLP_Autoencoder_01.pt"
    model.load_state_dict(torch.load(weights_path))
    model.to("cuda")
    model.train(True)

    train(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        learning_rate_end=LEARNING_RATE_END,
        batch_size=BATCH_SIZE,
        min_epochs=MIN_EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        model=model,
        name=NAME,
        use_wandb=False,
        patience=PATIENCE,
        predict_func=predict_func,
        save_best_model=True,
    )

# DiamondNet - Default weights, batchNorm, ReLU
# Epoch 50/100: 100% 265/265 [01:51<00:00,  2.38it/s, loss=95.3480, mse=95.3480, wmape=0.6455, mae=3.0573, val_loss=27.0051, val_mse=27.0051, val_wmape=0.7505, val_mae=0.9521]
# Test Accuracy: 69.2184

# Best DiamondFormer
# Test Accuracy: 85.8081