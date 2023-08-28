import sys; sys.path.append("../")

import torch
import torch.nn as nn
import torchmetrics

from utils import load_data, training_loop


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
    import os

    NUM_EPOCHS = 100
    MIN_EPOCHS = 50
    WARMUP_EPOCHS = 10
    PATIENCE = 20
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    NAME = "MAET05DIA"

    depths = [3, 3, 3, 3]
    dims = [32, 48, 64, 80]

    from model_MixerMLP import MLPMixer
    from model_Diamond import DiamondNet

    class BaseMLPMixer(nn.Module):
        def __init__(self, clamp_output=False, clamp_min=0.0, clamp_max=1.0):
            super(BaseMLPMixer, self).__init__()
            self.clamp_output = clamp_output
            self.clamp_min = clamp_min
            self.clamp_max = clamp_max
            
            self.model_foundation = MLPMixer(
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

            weights_folder = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/model_zoo/trained_models/"
            weights_path = os.path.join(weights_folder, "MLP_MaskedAutoencoder_01.pt")
            self.model_foundation.load_state_dict(torch.load(weights_path))

            for param in self.model_foundation.parameters():
                param.requires_grad = False

            self.model_finetune = DiamondNet(
                input_dim=self.model_foundation.stem_channels + 10,
                output_dim=1,
                input_size=64,
            )

            # self.model_finetune = MLPMixer(
            #     # chw=(self.model_foundation.stem_channels, 64, 64),
            #     chw=(10, 64, 64),
            #     output_dim=1,
            #     patch_size=4,
            #     embed_dim=256,
            #     expansion=4,
            #     drop_n=0.0,
            #     drop_p=0.0,
            #     dim=128,
            #     depth=3,
            # )

        def forward(self, identity):
            x = self.model_foundation.forward_stem(identity)
            x = self.model_foundation.forward_trunc(x)
            x = self.model_foundation.forward_reproject(x)
            # x = self.model_finetune.forward(x)
            # x = self.model_finetune.forward(identity)
            x = self.model_finetune.forward(torch.concat([x, identity], dim=1))

            if self.clamp_output:
                x = torch.clamp(x, 0.0, 100.0)
            
            return x

    model = BaseMLPMixer()
    model.to("cuda")

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