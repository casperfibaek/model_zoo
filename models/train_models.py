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
    save_best_model: bool = True,
    patience=20,
    min_epochs=50,
) -> str:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl_train, dl_val, dl_test = load_data(
        x="s2",
        y="area",
        with_augmentations=True,
        num_workers=8,
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
        criterion=TiledMSE(bias=0.5),
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
    import logging
    import os
    import sys

    # class StreamToLogger:
    #     def __init__(self, logger, log_level=logging.INFO):
    #         self.logger = logger
    #         self.log_level = log_level
    #         self.linebuf = ''

    #     def write(self, buf):
    #         for line in buf.rstrip().splitlines():
    #             self.logger.log(self.log_level, line.rstrip())

    #     def flush(self):
    #         pass

    NUM_EPOCHS = 50
    MIN_EPOCHS = 25
    WARMUP_EPOCHS = 10
    PATIENCE = 20
    LEARNING_RATE = 0.001
    LEARNING_RATE_END = 0.00001
    BATCH_SIZE = 16
    NAME = "MixerNano07"

    # log_format = '%(asctime)s - %(message)s'
    # logging.basicConfig(filename=os.path.join("../logs/", f"{NAME}.log"), level=logging.INFO, format=log_format)
    # logger = logging.getLogger()

    # stdout_logger = logging.getLogger('STDOUT')
    # sl = StreamToLogger(stdout_logger, logging.INFO)
    # sys.stdout = sl

    # stderr_logger = logging.getLogger('STDERR')
    # sl = StreamToLogger(stderr_logger, logging.ERROR)
    # sys.stderr = sl

    # import gc; gc.collect()
    # torch.cuda.empty_cache()

    # from model_CoreCNN_versions import CoreUnet_nano
    # model = CoreUnet_nano(
    #     input_dim=10,
    #     output_dim=1,
    # )


    from models.model_Mixer_versions import Mixer_nano
    model = Mixer_nano(
        chw=(10, 64, 64),
        output_dim=1,
        drop_n=0.1,
        drop_p=0.1,
    )
    # model = torch.compile(model)

    # from model_VisionTransformer import ViT
    # model = ViT(
    #     chw=(10, 64, 64),
    #     output_dim=1,
    #     patch_size=4,
    #     embed_dim=768,
    #     depth=3,
    #     num_heads=16,
    # )

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

# DiamondNet best result
# Test Accuracy: 67.6795

# MixerMLP best result
# Test Accuracy: 73.8183
