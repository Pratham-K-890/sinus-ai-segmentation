import numpy as np
import tensorflow as tf
import nrrd
from scipy.ndimage import zoom, label


MODEL_PATH = r"models/3dunet.h5"
TARGET_SHAPE = (64, 128, 128)


def resize_3d(vol, target, is_label=False):

    if vol.ndim == 4:
        vol = vol[..., 0]

    if vol.ndim == 2:
        vol = vol[np.newaxis, :, :]

    factors = (
        target[0] / vol.shape[0],
        target[1] / vol.shape[1],
        target[2] / vol.shape[2],
    )

    return zoom(vol, factors, order=0 if is_label else 1)


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter = tf.reduce_sum(y_true * y_pred, axis=1)
    union = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1)
    return tf.reduce_mean((2.*inter + smooth) / (union + smooth))


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return dice_loss(y_true, y_pred) + bce


def iou(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
    return (inter + 1e-6) / (union + 1e-6)


def keep_largest_component(mask):

    labeled, num = label(mask)

    if num == 0:
        return mask

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    largest_label = sizes.argmax()

    clean_mask = (labeled == largest_label).astype(np.uint8)

    return clean_mask


def run_segmentation(INPUT_PATH, OUTPUT_PATH):

    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "combined_loss": combined_loss,
            "dice_coef": dice_coef,
            "iou": iou
        },
        compile=False
    )

    print("Model loaded successfully")

    print("Reading input NRRD...")

    img, header = nrrd.read(INPUT_PATH)
    original_shape = img.shape

    img = np.clip(img, -1000, 1000)

    mean = img.mean()
    std = img.std()
    if std == 0:
        std = 1

    img = (img - mean) / std

    img_resized = resize_3d(img, TARGET_SHAPE)
    img_resized = img_resized[..., np.newaxis]
    img_resized = np.expand_dims(img_resized, axis=0).astype(np.float32)

    print("Running segmentation...")
    pred = model.predict(img_resized, verbose=1)[0, ..., 0]
    pred = (pred > 0.5).astype(np.uint8)

    pred = keep_largest_component(pred)

    pred_resized = resize_3d(pred, original_shape, is_label=True)

    nrrd.write(
        OUTPUT_PATH,
        pred_resized.astype(np.uint8),
        header
    )

    print("Segmentation saved at:")
    print(OUTPUT_PATH)

    return OUTPUT_PATH
