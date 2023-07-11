import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import buteo as beo
from glob import glob
from tqdm import tqdm
from functools import cmp_to_key

# NOTE: To run this you need ffmpeg installed. `conda install -c conda-forge ffmpeg`

DIVISIONS = 30
DPI = 130
HEIGHT = 1920
WIDTH = 1080
LIMIT = -1
PRERUN = 3
TEXT_COLOR = "#BBBBBB"
IMG_RAMP = "magma"
IMG_BACKGROUND = "#0f0f0f"
SQR_1 = "#FFC300"
SQR_2 = "#C70039"
LOC = 2
FOLDER = "../visualisations/"
IMG_GLOB = f"{FOLDER}pred_PyramidFemto3_*.tif"
VMIN = 0.0
VMAX = 100.0 
THRESHOLD = 1.0 # Minumum value to be considered a prediction (0.01 for [0, 1] and 1.0 for [0, 100])

places = [
    {
        "id": "0",
        "name": "Naestved",
        "name_abr": "naeastved",
        "sqr1": [700, 350],
        "sqr2": [1000, 1650],
        "rgb": "../data/images/naestved_s2.tif",
        "out_path": "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/model_zoo/visualisations/pred_Diamond3_naestved.mp4",
    },
]


def cmp_items(a, b):
    a_int = int(os.path.splitext(os.path.basename(a))[0].split("_")[LOC])
    b_int = int(os.path.splitext(os.path.basename(b))[0].split("_")[LOC])
    if a_int > b_int:
        return 1
    elif a_int == b_int:
        return 0
    else:
        return -1

def ready_image(glob_path, divisions=10, limit=-1, smooth=3):
    images = glob(glob_path)
    images.sort(key=cmp_to_key(cmp_items))

    if limit == -1:
        limit = len(images)

    arrays = []
    for img in images[:limit]:
        img = beo.raster_to_array(img)[:, :, 0]
        arrays.append(img)

    ret_arr = None

    arrays_interpolated = []
    for idx, arr in enumerate(arrays):
        arr_from = arrays[idx-1] if idx != 0 else np.zeros_like(arr)

        interpolated = np.linspace(arr_from, arr, divisions, axis=0)
        for i in range(divisions):
            arrays_interpolated.append(interpolated[i, :, :])
    
    if smooth != 0:
        arrays_smoothed = []
        for idx, arr in enumerate(arrays_interpolated):
            to_smooth = []
            if idx < smooth:
                for i in range(smooth):
                    to_smooth.append(arrays_interpolated[idx + i])
                
            elif idx > len(arrays_interpolated) - smooth:
                for i in range(smooth):
                    to_smooth.append(arrays_interpolated[idx - i])
            else:
                for i in range(smooth):
                    if i == 0:
                        to_smooth.append(arrays_interpolated[idx])
                    else:
                        to_smooth.append(arrays_interpolated[idx + i])
                        to_smooth.append(arrays_interpolated[idx - i])

            smoothed = np.mean(to_smooth, axis=0)
            arrays_smoothed.append(smoothed)

        ret_arr = np.array(arrays_smoothed)
    
    else:
        ret_arr = np.array(arrays_interpolated)

    return ret_arr


for idx, place in enumerate(places):
    fig_id = place["id"]
    fig_name = place["name"]
    fig_name_abr = place["name_abr"]
    fig_out_path = place["out_path"]

    print(f"Reading: {fig_name}")

    fig = plt.figure(figsize=(HEIGHT / DPI, WIDTH / DPI), dpi=DPI, facecolor=IMG_BACKGROUND)

    spec = gridspec.GridSpec(nrows=2, ncols=3)

    ax_main = fig.add_subplot(spec[0:2, 0:2])
    ax_sup1 = fig.add_subplot(spec[0, 2])
    ax_sup2 = fig.add_subplot(spec[1, 2])

    ax_main.set_axis_off(); ax_main.set_xticks([]); ax_main.set_yticks([])
    ax_sup1.set_axis_off(); ax_sup1.set_xticks([]); ax_sup1.set_yticks([])
    ax_sup2.set_axis_off(); ax_sup2.set_xticks([]); ax_sup2.set_yticks([])

    ax_main.set_facecolor(IMG_BACKGROUND)
    ax_sup1.set_facecolor(IMG_BACKGROUND)
    ax_sup2.set_facecolor(IMG_BACKGROUND)

    size = 500
    sup1_rect = place["sqr1"]
    sup2_rect = place["sqr2"]

    lw = 3
    rect1 = patches.Rectangle((sup1_rect[0], sup1_rect[1]), size, size, linewidth=lw, edgecolor=SQR_1, facecolor='none')
    rect2 = patches.Rectangle((sup2_rect[0], sup2_rect[1]), size, size, linewidth=lw, edgecolor=SQR_2, facecolor='none')

    ax_sqr1_border = patches.Rectangle((lw / 2, lw / 2), size - (lw + (lw / 2)), size - (lw + (lw / 2)), linewidth=lw, edgecolor=SQR_1, facecolor='none')
    ax_sqr2_border = patches.Rectangle((lw / 2, lw / 2), size - (lw + (lw / 2)), size - (lw + (lw / 2)), linewidth=lw, edgecolor=SQR_2, facecolor='none')

    rgb_image = place["rgb"]
    rgb_image = beo.raster_to_array(rgb_image, filled=True, fill_value=0.0, cast=np.float32, bands=[3, 2, 1])

    q02 = np.quantile(rgb_image, 0.02)
    q99 = np.quantile(rgb_image, 0.98)

    rgb_image = np.clip(rgb_image, q02, q99)
    rgb_image = (rgb_image - q02) / (q99 - q02)

    predictions = ready_image(IMG_GLOB, divisions=DIVISIONS, limit=LIMIT)
    print(f"Read: {fig_name}")

    sqr1_data_rgb = rgb_image[sup1_rect[1]:sup1_rect[1]+size, sup1_rect[0]:sup1_rect[0]+size, :]
    sqr1_data_pred = predictions[0, sup1_rect[1]:sup1_rect[1] + size, sup1_rect[0]:sup1_rect[0] + size]

    sqr2_data_rgb = rgb_image[sup2_rect[1]:sup2_rect[1]+size, sup2_rect[0]:sup2_rect[0]+size, :]
    sqr2_data_pred = predictions[0, sup2_rect[1]:sup2_rect[1] + size, sup2_rect[0]:sup2_rect[0] + size]

    # Main prediction window
    main_rgb = ax_main.imshow(rgb_image, interpolation="antialiased")
    ax_main.add_patch(rect1)
    ax_main.add_patch(rect2)
    main_pred = ax_main.imshow(predictions[0], vmin=VMIN, vmax=VMAX, interpolation="antialiased", cmap=IMG_RAMP, alpha=np.zeros_like(predictions[0]))

    # First Square
    sqr1_rgb = ax_sup1.imshow(sqr1_data_rgb, interpolation="antialiased")
    ax_sup1.add_patch(ax_sqr1_border)
    sqr1_pred = ax_sup1.imshow(sqr1_data_pred, vmin=VMIN, vmax=VMAX, interpolation="antialiased", cmap=IMG_RAMP, alpha=np.zeros_like(sqr1_data_pred))

    # Second Square
    sqr2_rgb = ax_sup2.imshow(sqr2_data_rgb, interpolation="antialiased")
    ax_sup2.add_patch(ax_sqr2_border)
    sqr2_pred = ax_sup2.imshow(sqr2_data_pred, vmin=VMIN, vmax=VMAX, interpolation="antialiased", cmap=IMG_RAMP, alpha=np.zeros_like(sqr2_data_pred))

    warm_up = DIVISIONS * PRERUN
    times = np.array([0.0] * warm_up + list(np.arange(1, len(predictions) + 1)), dtype=np.float32) / DIVISIONS

    time_text = ax_main.text(0.05, 0.05, str(round(times[0], 1)), fontsize=15, color=TEXT_COLOR, transform=ax_main.transAxes)
    place_text = ax_main.text(0.5, 0.95, fig_name, fontsize=15, color=TEXT_COLOR, transform=ax_main.transAxes)

    plt.tight_layout()
    fig.subplots_adjust(wspace=-0.345, hspace=0.01, left=0.0, right=1.0, top=1.0, bottom=0.0)
    plt.rcParams["figure.facecolor"] = IMG_BACKGROUND

    total_length = len(predictions) + warm_up
    assert total_length == len(times), "Times and predictions are not the same length"

    tqbar = tqdm(total=total_length, desc=f"Saving: {fig_name}", position=0, leave=True)
    def updatefig(j):
        global ax_main, ax_sup1, ax_sup2, rgb_image, predictions, sqr1_rgb, sqr2_rgb, sup1_rect, sup1_rect, sqr1_pred, sqr2_pred, time_text, place_text, THRESHOLD

        if j < warm_up:
            main_rgb.set_data(rgb_image)
            sqr1_rgb.set_data(sqr1_data_rgb)
            sqr2_rgb.set_data(sqr2_data_rgb)

            return [main_rgb, main_pred, sqr1_rgb, sqr1_pred, sqr2_rgb, sqr2_pred]
        
        idx = j - warm_up
        pred = predictions[idx, :, :]

        main_rgb.set_data(np.clip(rgb_image, 0.0, 1.0))
        main_pred.set_data(pred)
        main_pred.set_alpha(np.where(pred > THRESHOLD, 1.0, pred / THRESHOLD).astype(np.float32))

        sqr1_rgb.set_data(np.clip(sqr1_data_rgb, 0.0, 1.0))
        sqr1_data_pred = pred[sup1_rect[1]:sup1_rect[1] + size, sup1_rect[0]:sup1_rect[0] + size]
        sqr1_pred.set_data(sqr1_data_pred)
        sqr1_pred.set_alpha(np.where(sqr1_data_pred > THRESHOLD, 1.0, sqr1_data_pred / THRESHOLD).astype(np.float32))

        sqr2_rgb.set_data(np.clip(sqr2_data_rgb, 0.0, 1.0))
        sqr2_data_pred = pred[sup2_rect[1]:sup2_rect[1] + size, sup2_rect[0]:sup2_rect[0] + size]
        sqr2_pred.set_data(sqr2_data_pred)
        sqr2_pred.set_alpha(np.where(sqr2_data_pred > THRESHOLD, 1.0, sqr2_data_pred / THRESHOLD).astype(np.float32))

        time_text.set_text(str(round(times[j], 1)))

        return [main_rgb, main_pred, sqr1_rgb, sqr1_pred, sqr2_rgb, sqr2_pred]

    anim = animation.FuncAnimation(
        fig,
        updatefig,
        frames=range(len(times)),
        interval=30,
        blit=True,
    )
    anim.save(
        fig_out_path,
        writer=animation.FFMpegWriter(fps=30, bitrate=1000000),
        savefig_kwargs={"facecolor": IMG_BACKGROUND},
        progress_callback=lambda _x, _y: tqbar.update(),
    )
    print(f"Saved: {fig_name}")
