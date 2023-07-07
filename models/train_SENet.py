import sys; sys.path.append("../")

import torch
import torch.nn as nn
import torchmetrics

from utils import load_data, training_loop, TiledMSE
from model_SENet import SENet_femto, SENetUnet_femto


def train(
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_init_std: float = 0.02,
    activation: str = "gelu",
    warmup_epochs: int = 10,
    name=str,
    use_wandb: bool = True,
    predict_func=None,
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

    input_shape = dl_train.dataset[0][0].shape
    input_dim = input_shape[0]
    input_height = input_shape[1]
    input_width = input_shape[2]

    model = SENetUnet_femto(input_dim=input_dim, output_dim=1, clamp_output=True, clamp_min=0.0, clamp_max=100.0, activation=activation)
    model.initialize_weights(weight_init_std)
    model(torch.randn((batch_size, input_dim, input_height, input_width)))

    wmape = torchmetrics.WeightedMeanAbsolutePercentageError(); wmape.__name__ = "wmape"
    mae = torchmetrics.MeanAbsoluteError(); mae.__name__ = "mae"
    mse = torchmetrics.MeanSquaredError(); mse.__name__ = "mse"

    training_loop(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
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
    import wandb

    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    NAME = "SENetFemto"

    # train(
    #     num_epochs=NUM_EPOCHS,
    #     learning_rate=LEARNING_RATE,
    #     batch_size=BATCH_SIZE,
    #     name=NAME,
    #     use_wandb=False,
    # )

    # exit()

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'   
        },
        'parameters': {
            'learning_rate': {
                "values": [0.01, 0.001, 0.0001],
            },
            'batch_size': {
                'values': [8, 16, 32, 64, 128],
            },
            "weight_init_std": {
                "values": [0.01, 0.02, 0.05, 0.1, 0.2],
            },
            "activation": {
                "values": ["relu", "gelu"],
            },
            "warmup_epochs":{
                "values": [0, 5, 10],
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config)

    def _train():
        with wandb.init():
            config = wandb.config

            train(
                num_epochs=NUM_EPOCHS,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                weight_init_std=config.weight_init_std,
                activation=config.activation,
                warmup_epochs=config.warmup_epochs,
                name=NAME,
                # predict_func=predict_func,
                predict_func=None,
                use_wandb=True,
            )

    wandb.agent(sweep_id, function=_train, project="model_zoo")
