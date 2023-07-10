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
    patience=10,
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
        criterion=TiledMSE(0.5),
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
    from model_BasicCNN import BasicUnet_femto, BasicUnet_base

    # torch.autograd.set_detect_anomaly(True)

    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NAME = "BasicCNNFemto"

    model = BasicUnet_femto(input_dim=10, output_dim=1, clamp_output=True, clamp_min=0.0, clamp_max=100.0, activation="relu")
    model.initialize_weights(0.05)

    train(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        model=model,
        name=NAME,
        use_wandb=False,
        patience=50,
        # predict_func=predict_func,
    )
