import sys; sys.path.append("../")

import torch
import torch.nn as nn
import torchmetrics

from utils import load_data, training_loop
from model_ConvNextV2 import ConvNextV2_femto, ConvNextV2Unet_femto


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

    model = ConvNextV2Unet_femto(input_dim=input_dim, output_dim=1, clamp_output=True, clamp_min=0.0, clamp_max=100.0)
    model.initialize_weights()
    model(torch.randn((batch_size, input_dim, input_height, input_width)))

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
        batch_size=batch_size,
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
        out_path=F"../visualisations/pred_ConvNextV2-2_{epoch}.tif",
    )


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=UserWarning)
    import buteo as beo
    import numpy as np

    LEARNING_RATE = 0.001
    NUM_EPOCHS = 300
    BATCH_SIZE = 32
    NAME = "ConvNextV2-2"

    train(
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        name=NAME,
        predict_func=predict_func,
    )

# Epoch 2/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:46<00:00,  2.85it/s, loss=188.3584, mse=188.3585, wmape=1.1532, mae=5.3625, val_loss=53.9551, val_mse=53.9551, val_wmape=1.3245, val_mae=1.7572]
# Epoch 3/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:33<00:00,  3.95it/s, loss=180.1477, mse=180.1476, wmape=1.1186, mae=5.1644, val_loss=61.2032, val_mse=61.2033, val_wmape=1.0440, val_mae=1.4206]
# Epoch 4/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:58<00:00,  2.27it/s, loss=174.4083, mse=174.4083, wmape=1.0907, mae=4.9855, val_loss=51.6236, val_mse=51.6236, val_wmape=1.2998, val_mae=1.7181]
# Epoch 5/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [01:00<00:00,  2.18it/s, loss=162.2027, mse=162.2028, wmape=1.0073, mae=4.6983, val_loss=51.5600, val_mse=51.5600, val_wmape=0.9943, val_mae=1.3465]
# Epoch 6/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:55<00:00,  2.40it/s, loss=154.5067, mse=154.5067, wmape=0.9572, mae=4.5017, val_loss=45.1683, val_mse=45.1683, val_wmape=1.0244, val_mae=1.3818]
# Epoch 7/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [01:08<00:00,  1.94it/s, loss=146.2150, mse=146.2151, wmape=0.9217, mae=4.3240, val_loss=43.6171, val_mse=43.6171, val_wmape=1.0754, val_mae=1.4386]
# Epoch 8/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:35<00:00,  3.68it/s, loss=156.3263, mse=156.3263, wmape=0.9698, mae=4.5065, val_loss=52.5607, val_mse=52.5607, val_wmape=1.5728, val_mae=2.0381]
# Epoch 9/300: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:36<00:00,  3.59it/s, loss=155.1622, mse=155.1622, wmape=0.9588, mae=4.5100, val_loss=53.7869, val_mse=53.7869, val_wmape=2.0131, val_mae=2.5262]
# Epoch 10/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:37<00:00,  3.55it/s, loss=147.8872, mse=147.8873, wmape=0.9233, mae=4.3343, val_loss=45.4618, val_mse=45.4618, val_wmape=1.1635, val_mae=1.5509]
# Epoch 11/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:57<00:00,  2.29it/s, loss=143.6942, mse=143.6942, wmape=0.9091, mae=4.2275, val_loss=41.5100, val_mse=41.5100, val_wmape=1.1036, val_mae=1.4676]
# Epoch 12/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:55<00:00,  2.38it/s, loss=138.3908, mse=138.3908, wmape=0.8790, mae=4.1142, val_loss=40.3668, val_mse=40.3668, val_wmape=1.0887, val_mae=1.4510]
# Epoch 13/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:34<00:00,  3.87it/s, loss=133.7183, mse=133.7183, wmape=0.8529, mae=4.0116, val_loss=41.6810, val_mse=41.6810, val_wmape=0.8789, val_mae=1.1939]
# Epoch 14/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:33<00:00,  3.93it/s, loss=128.9058, mse=128.9058, wmape=0.8289, mae=3.9069, val_loss=42.4050, val_mse=42.4050, val_wmape=0.8301, val_mae=1.1296]
# Epoch 15/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:57<00:00,  2.29it/s, loss=125.7604, mse=125.7605, wmape=0.8225, mae=3.8396, val_loss=37.3617, val_mse=37.3617, val_wmape=0.9001, val_mae=1.2156]
# Epoch 16/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:34<00:00,  3.83it/s, loss=123.8944, mse=123.8944, wmape=0.8125, mae=3.8080, val_loss=38.1950, val_mse=38.1950, val_wmape=0.8641, val_mae=1.1729]
# Epoch 17/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:33<00:00,  3.98it/s, loss=122.5958, mse=122.5958, wmape=0.8016, mae=3.7851, val_loss=41.7706, val_mse=41.7706, val_wmape=0.8581, val_mae=1.1643]
# Epoch 18/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:34<00:00,  3.77it/s, loss=138.8871, mse=138.8871, wmape=0.8816, mae=4.1001, val_loss=47.1478, val_mse=47.1478, val_wmape=0.9688, val_mae=1.3050]
# Epoch 19/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:33<00:00,  3.95it/s, loss=136.3180, mse=136.3180, wmape=0.8547, mae=4.0456, val_loss=120.689, val_mse=120.689, val_wmape=2.1060, val_mae=2.7787]
# Epoch 20/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:33<00:00,  3.95it/s, loss=132.5231, mse=132.5231, wmape=0.8471, mae=3.9664, val_loss=46.2678, val_mse=46.2678, val_wmape=0.9155, val_mae=1.2466]
# Epoch 21/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:35<00:00,  3.76it/s, loss=130.7646, mse=130.7646, wmape=0.8315, mae=3.9178, val_loss=50.4864, val_mse=50.4864, val_wmape=0.9157, val_mae=1.2480]
# Epoch 22/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:34<00:00,  3.86it/s, loss=127.6350, mse=127.6349, wmape=0.8296, mae=3.8620, val_loss=42.7200, val_mse=42.7200, val_wmape=0.9008, val_mae=1.2192]
# Epoch 23/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:34<00:00,  3.81it/s, loss=126.1348, mse=126.1348, wmape=0.8096, mae=3.8072, val_loss=47.9735, val_mse=47.9735, val_wmape=0.8773, val_mae=1.1915]
# Epoch 24/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:36<00:00,  3.58it/s, loss=123.2228, mse=123.2227, wmape=0.7874, mae=3.7399, val_loss=45.6570, val_mse=45.6570, val_wmape=1.0913, val_mae=1.4683]
# Epoch 25/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [01:02<00:00,  2.13it/s, loss=120.8168, mse=120.8168, wmape=0.7801, mae=3.6966, val_loss=34.2843, val_mse=34.2843, val_wmape=0.8072, val_mae=1.0991]
# Epoch 26/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:40<00:00,  3.27it/s, loss=119.8770, mse=119.8770, wmape=0.7710, mae=3.6878, val_loss=47.1676, val_mse=47.1676, val_wmape=0.8780, val_mae=1.1860]
# Epoch 27/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:34<00:00,  3.84it/s, loss=117.1315, mse=117.1314, wmape=0.7647, mae=3.6051, val_loss=34.6760, val_mse=34.6760, val_wmape=0.8048, val_mae=1.0933]
# Epoch 28/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [01:06<00:00,  1.98it/s, loss=114.5842, mse=114.5842, wmape=0.7489, mae=3.5470, val_loss=33.4792, val_mse=33.4792, val_wmape=0.8110, val_mae=1.0961]
# Epoch 29/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:36<00:00,  3.62it/s, loss=113.8331, mse=113.8331, wmape=0.7407, mae=3.5243, val_loss=38.9155, val_mse=38.9155, val_wmape=0.8751, val_mae=1.1873]
# Epoch 30/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [01:02<00:00,  2.11it/s, loss=112.5733, mse=112.5733, wmape=0.7413, mae=3.5066, val_loss=33.4563, val_mse=33.4563, val_wmape=0.7884, val_mae=1.0701]
# Epoch 31/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [01:00<00:00,  2.20it/s, loss=112.7130, mse=112.7130, wmape=0.7377, mae=3.5080, val_loss=33.4005, val_mse=33.4005, val_wmape=0.7867, val_mae=1.0658]
# Epoch 32/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:33<00:00,  3.95it/s, loss=112.2276, mse=112.2276, wmape=0.7397, mae=3.5167, val_loss=34.6201, val_mse=34.6201, val_wmape=0.8040, val_mae=1.0893]
# Epoch 33/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:55<00:00,  2.37it/s, loss=109.6871, mse=109.6871, wmape=0.7209, mae=3.4474, val_loss=32.3366, val_mse=32.3366, val_wmape=0.7961, val_mae=1.0796]
# Epoch 34/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:37<00:00,  3.57it/s, loss=110.2537, mse=110.2537, wmape=0.7318, mae=3.4641, val_loss=32.6301, val_mse=32.6301, val_wmape=0.7616, val_mae=1.0313]
# Epoch 35/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:56<00:00,  2.33it/s, loss=109.8789, mse=109.8789, wmape=0.7218, mae=3.4332, val_loss=32.5939, val_mse=32.5939, val_wmape=0.7739, val_mae=1.0493]
# Epoch 36/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [01:01<00:00,  2.16it/s, loss=107.3108, mse=107.3108, wmape=0.7244, mae=3.3967, val_loss=32.2726, val_mse=32.2726, val_wmape=0.7794, val_mae=1.0558]
# Epoch 37/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:35<00:00,  3.68it/s, loss=118.8361, mse=118.8361, wmape=0.7746, mae=3.6383, val_loss=36.4987, val_mse=36.4988, val_wmape=0.7998, val_mae=1.0861]
# Epoch 38/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:40<00:00,  3.28it/s, loss=137.4635, mse=137.4635, wmape=0.8686, mae=4.0594, val_loss=63.1813, val_mse=63.1813, val_wmape=0.9108, val_mae=1.2521]
# Epoch 39/300: 100%|███████████████████████████████████████████████████████████████████████████| 132/132 [00:39<00:00,  3.34it/s, loss=128.0619, mse=128.0618, wmape=0.8187, mae=3.8322, val_loss=37.4181, val_mse=37.4181, val_wmape=0.8777, val_mae=1.1718]