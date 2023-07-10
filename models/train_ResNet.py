import sys; sys.path.append("../")

import torch
import torch.nn as nn
import torchmetrics

from utils import load_data, training_loop
from model_ResNet import ResNet_femto, ResNetUnet_femto


def train(
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    name=str,
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

    model = ResNetUnet_femto(input_dim=input_dim, output_dim=1, clamp_output=True, clamp_min=0.0, clamp_max=100.0)
    model.initialize_weights()

    wmape = torchmetrics.WeightedMeanAbsolutePercentageError(); wmape.__name__ = "wmape"
    mae = torchmetrics.MeanAbsoluteError(); mae.__name__ = "mae"

    training_loop(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        model=model,
        criterion=nn.MSELoss(),
        device=device,
        metrics=[
            wmape.to(device),
            mae.to(device),
        ],
        train_loader=dl_train,
        val_loader=dl_val,
        test_loader=dl_test,
        name=name,
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
            batch_size=BATCH_SIZE,
        )
    beo.array_to_raster(
        predicted,
        reference=img_path,
        out_path=F"../visualisations/pred_ResNetFemtoV2_{epoch}.tif",
    )


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=UserWarning)
    import buteo as beo
    import numpy as np

    LEARNING_RATE = 0.001
    NUM_EPOCHS = 300
    BATCH_SIZE = 32
    NAME = "ResNetFemtoV2"

    train(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        name=NAME,
        predict_func=predict_func,
    )


# Epoch 1/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:42<00:00,  3.11it/s, loss=219.2646, mse=219.2646, wmape=1.2220, mae=5.6619, val_loss=68.3760, val_mse=68.3760, val_wmape=2.4376, val_mae=3.0202]
# Epoch 2/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:47<00:00,  2.80it/s, loss=180.5357, mse=180.5356, wmape=1.1182, mae=5.1758, val_loss=52.2494, val_mse=52.2494, val_wmape=1.2773, val_mae=1.6928]
# Epoch 3/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:28<00:00,  4.57it/s, loss=175.9384, mse=175.9384, wmape=1.0906, mae=5.0467, val_loss=56.9945, val_mse=56.9945, val_wmape=1.3822, val_mae=1.8161]
# Epoch 4/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:51<00:00,  2.57it/s, loss=169.7944, mse=169.7944, wmape=1.0561, mae=4.8570, val_loss=48.7446, val_mse=48.7446, val_wmape=1.1752, val_mae=1.5598]
# Epoch 5/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:36<00:00,  3.59it/s, loss=158.6714, mse=158.6714, wmape=0.9921, mae=4.5886, val_loss=57.5529, val_mse=57.5529, val_wmape=0.9223, val_mae=1.2675]
# Epoch 6/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [01:01<00:00,  2.13it/s, loss=153.1731, mse=153.1731, wmape=0.9591, mae=4.4774, val_loss=44.1910, val_mse=44.1910, val_wmape=0.9607, val_mae=1.3025]
# Epoch 7/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [01:00<00:00,  2.19it/s, loss=146.1304, mse=146.1304, wmape=0.9220, mae=4.3168, val_loss=43.2198, val_mse=43.2199, val_wmape=1.0310, val_mae=1.3898]
# Epoch 8/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:31<00:00,  4.23it/s, loss=156.2081, mse=156.2080, wmape=0.9765, mae=4.4905, val_loss=67.6771, val_mse=67.6771, val_wmape=1.3207, val_mae=1.8109]
# Epoch 9/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:30<00:00,  4.36it/s, loss=151.5307, mse=151.5307, wmape=0.9414, mae=4.4058, val_loss=53.9794, val_mse=53.9794, val_wmape=1.2280, val_mae=1.6496]
# Epoch 10/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:32<00:00,  4.10it/s, loss=148.6740, mse=148.6741, wmape=0.9210, mae=4.3299, val_loss=44.9311, val_mse=44.9311, val_wmape=0.9549, val_mae=1.2988]
# Epoch 11/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:31<00:00,  4.17it/s, loss=142.8532, mse=142.8531, wmape=0.9013, mae=4.1956, val_loss=73.2996, val_mse=73.2996, val_wmape=2.0158, val_mae=2.5413]