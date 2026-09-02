import SimpleITK as sitk
import nrrd
import numpy as np
from scipy.ndimage import label, generate_binary_structure

MIN_VOXELS = 500


def compute_metrics(component_mask, spacing):

    coords = np.argwhere(component_mask)

    if coords.size == 0:
        return None

    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    length  = (x_max - x_min + 1) * spacing[0]
    breadth = (y_max - y_min + 1) * spacing[1]
    height  = (z_max - z_min + 1) * spacing[2]

    volume = coords.shape[0] * voxel_volume

    return volume, length, breadth, height


def extract_valid_components(mask):

    structure = generate_binary_structure(3, 3)
    labeled, num = label(mask, structure=structure)

    components = []

    for i in range(1, num + 1):
        voxel_count = np.sum(labeled == i)
        if voxel_count >= MIN_VOXELS:
            components.append((i, voxel_count))

    return labeled, components


def centroid_x(component):
    coords = np.argwhere(component)
    return coords[:, 2].mean()


def run_metrics(INPUT_SEG):

    image = sitk.ReadImage(INPUT_SEG)
    spacing = image.GetSpacing()

    mask = sitk.GetArrayFromImage(image)
    mask = mask > 0

    labeled, components = extract_valid_components(mask)

    print("Valid components found:", len(components))

    if len(components) == 0:
        print("No sinus detected.")
        return None, None

    components.sort(key=lambda x: x[1], reverse=True)

    if len(components) == 1:
        comp = labeled == components[0][0]
        metrics = compute_metrics(comp, spacing)
        return metrics, None

    comp1 = labeled == components[0][0]
    comp2 = labeled == components[1][0]

    if centroid_x(comp1) < centroid_x(comp2):
        left_mask = comp1
        right_mask = comp2
    else:
        left_mask = comp2
        right_mask = comp1

    left_metrics = compute_metrics(left_mask, spacing)
    right_metrics = compute_metrics(right_mask, spacing)

    data, header = nrrd.read(INPUT_SEG)
    print("\nVoxel spacing from header:")
    print(header["space directions"])

    return left_metrics, right_metrics
