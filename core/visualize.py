import streamlit as st
import SimpleITK as sitk
import numpy as np
import base64
import io
from PIL import Image
import hashlib
import uuid


# ================= HASH FILE =================
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ================= CT STACK =================
def ct_to_png_stack(volume):

    images = []
    vmin, vmax = np.percentile(volume, (1, 99))

    for i in range(volume.shape[0]):
        slice_img = volume[i]
        slice_img = np.clip(slice_img, vmin, vmax)
        slice_img = (slice_img - vmin) / (vmax - vmin)
        slice_img = (slice_img * 255).astype(np.uint8)

        pil = Image.fromarray(slice_img).convert("L")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")

        images.append(base64.b64encode(buf.getvalue()).decode())

    return images


# ================= SEGMENT STACK =================
def mask_to_color_stack(mask):

    images = []

    for i in range(mask.shape[0]):
        m = mask[i]
        rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)

        # red transparent mask
        rgba[m > 0] = [255, 0, 0, 120]

        pil = Image.fromarray(rgba, mode="RGBA")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")

        images.append(base64.b64encode(buf.getvalue()).decode())

    return images


# ================= CACHE WITH FILE HASH =================
@st.cache_data(show_spinner=False)
def prepare_stack(nrrd_path, seg_path, hash1, hash2):

    img = sitk.GetArrayFromImage(sitk.ReadImage(nrrd_path))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

    img_stack = ct_to_png_stack(img)
    seg_stack = mask_to_color_stack(seg)

    return img_stack, seg_stack, img.shape[0]


# ================= MAIN VIEWER =================
def show_slices(nrrd_path, seg_path):

    # force new viewer each run
    viewer_id = str(uuid.uuid4()).replace("-", "")

    h1 = file_hash(nrrd_path)
    h2 = file_hash(seg_path)

    img_stack, seg_stack, depth = prepare_stack(nrrd_path, seg_path, h1, h2)

    html = f"""
    <div style="text-align:center">

        <h4>CBCT Sinus Viewer</h4>

        <canvas id="viewer_{viewer_id}" width="512" height="512"
        style="border:2px solid #444; cursor:grab;"></canvas>

        <br><br>

        <input type="range" min="0" max="{depth-1}" value="{depth//2}"
        id="slider_{viewer_id}" style="width:520px">

        <br><br>

        <button onclick="play_{viewer_id}()">▶ Play</button>
        <button onclick="stop_{viewer_id}()">⏸ Stop</button>

        <p id="sliceLabel_{viewer_id}"></p>

    </div>

    <script>

    const imgStack = {img_stack};
    const segStack = {seg_stack};

    const canvas = document.getElementById("viewer_{viewer_id}");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("slider_{viewer_id}");
    const label = document.getElementById("sliceLabel_{viewer_id}");

    let current = parseInt(slider.value);
    let interval = null;

    function drawSlice_{viewer_id}(index){{
        current = index;
        slider.value = index;
        label.innerHTML = "Slice: " + index;

        ctx.clearRect(0,0,512,512);

        let img = new Image();
        img.src = "data:image/png;base64," + imgStack[index];
        img.onload = function(){{
            ctx.drawImage(img,0,0,512,512);

            let seg = new Image();
            seg.src = "data:image/png;base64," + segStack[index];
            seg.onload = function(){{
                ctx.drawImage(seg,0,0,512,512);
            }};
        }};
    }}

    slider.oninput = function(){{
        drawSlice_{viewer_id}(parseInt(this.value));
    }}

    canvas.addEventListener("wheel", function(e){{
        e.preventDefault();

        if(e.deltaY < 0)
            current = Math.max(0, current - 1);
        else
            current = Math.min(imgStack.length - 1, current + 1);

        drawSlice_{viewer_id}(current);
    }});

    function play_{viewer_id}(){{
        if(interval) return;
        interval = setInterval(() => {{
            current = (current + 1) % imgStack.length;
            drawSlice_{viewer_id}(current);
        }}, 60);
    }}

    function stop_{viewer_id}(){{
        clearInterval(interval);
        interval = null;
    }}

    drawSlice_{viewer_id}(current);

    </script>
    """

    st.components.v1.html(html, height=750)
