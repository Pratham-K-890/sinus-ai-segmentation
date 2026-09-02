import os
import shutil
import SimpleITK as sitk
import pandas as pd
import numpy as np
from PIL import Image


# ================= SAVE NRRD DATA =================
def save_medical_volumes(nrrd_path, seg_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    # copy original volume
    vol_out = os.path.join(output_folder, "cbct_volume.nrrd")
    shutil.copy(nrrd_path, vol_out)

    # copy segmentation
    seg_out = os.path.join(output_folder, "sinus_segmentation.seg.nrrd")
    shutil.copy(seg_path, seg_out)

    return vol_out, seg_out


# ================= OPTIONAL PREVIEW IMAGE =================
def save_preview_image(nrrd_path, seg_path, output_folder):

    img = sitk.GetArrayFromImage(sitk.ReadImage(nrrd_path))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

    mid = img.shape[0] // 2

    ct = img[mid]
    mask = seg[mid]

    vmin, vmax = np.percentile(ct, (1, 99))
    ct = np.clip(ct, vmin, vmax)
    ct = ((ct - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    rgb = np.stack([ct, ct, ct], axis=-1)
    rgb[mask > 0] = [255, 0, 0]

    preview_path = os.path.join(output_folder, "preview_overlay.png")
    Image.fromarray(rgb).save(preview_path)

    return preview_path


def save_excel(left_metrics, right_metrics, gender, output_folder):

    data = {
        "Parameter": [
            "Left Volume", "Left Length", "Left Breadth", "Left Height",
            "Right Volume", "Right Length", "Right Breadth", "Right Height",
            "Predicted Gender"
        ],
        "Value": [
            *(left_metrics if left_metrics else [None]*4),
            *(right_metrics if right_metrics else [None]*4),
            gender
        ]
    }

    df = pd.DataFrame(data)

    excel_path = os.path.join(output_folder, "measurements.xlsx")
    df.to_excel(excel_path, index=False)

    return excel_path
