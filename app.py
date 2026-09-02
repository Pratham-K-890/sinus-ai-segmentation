import streamlit as st
import os
import uuid

from core.dicom_to_nrrd import convert_dicom_to_nrrd
from core.segment import run_segmentation
from core.metrics import run_metrics
from core.visualize import show_slices
from core.export import save_medical_volumes, save_preview_image, save_excel
from core.gender_prediction import predict_gender_from_volume


# ================= STATE INIT =================
if "nrrd_path" not in st.session_state:
    st.session_state.nrrd_path = None

if "seg_path" not in st.session_state:
    st.session_state.seg_path = None

if "left" not in st.session_state:
    st.session_state.left = None

if "right" not in st.session_state:
    st.session_state.right = None

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "gender" not in st.session_state:
    st.session_state.gender = None


st.set_page_config(layout="wide")
st.title("🦷 CBCT Sinus AI Analyzer")

mode = st.radio("Select Input Type", ["Upload DICOM Folder", "Upload NRRD File"])

os.makedirs("temp", exist_ok=True)


# ================= DICOM MODE =================
if mode == "Upload DICOM Folder":

    uploaded_files = st.file_uploader("Upload ALL DICOM files", accept_multiple_files=True)

    if uploaded_files:

        case_id = str(uuid.uuid4())[:8]
        case_folder = f"temp/{case_id}"
        os.makedirs(case_folder, exist_ok=True)

        dicom_folder = f"{case_folder}/dicom"
        os.makedirs(dicom_folder, exist_ok=True)

        for file in uploaded_files:
            with open(f"{dicom_folder}/{file.name}", "wb") as f:
                f.write(file.getbuffer())

        if st.button("🚀 Run AI"):

            with st.spinner("Converting DICOM → NRRD..."):
                st.session_state.nrrd_path = convert_dicom_to_nrrd(dicom_folder, case_folder)

            if not st.session_state.nrrd_path:
                st.error("DICOM conversion failed.")
                st.stop()

            seg_out = os.path.join(case_folder, "seg.seg.nrrd")

            with st.spinner("Running 3D AI Segmentation..."):
                st.session_state.seg_path = run_segmentation(st.session_state.nrrd_path, seg_out)

            with st.spinner("Calculating Sinus Metrics..."):
                st.session_state.left, st.session_state.right = run_metrics(st.session_state.seg_path)

            # ---------- RULE-BASED GENDER ----------
            total_volume = 0

            if st.session_state.left:
                total_volume += st.session_state.left[0]

            if st.session_state.right:
                total_volume += st.session_state.right[0]

            if total_volume > 0:
                st.session_state.gender = predict_gender_from_volume(total_volume)
            else:
                st.session_state.gender = None

            st.session_state.analysis_done = True
            st.success("Analysis Complete")


# ================= NRRD MODE =================
else:

    uploaded_nrrd = st.file_uploader("Upload NRRD volume", type=["nrrd"])

    if uploaded_nrrd:

        case_id = str(uuid.uuid4())[:8]
        case_folder = f"temp/{case_id}"
        os.makedirs(case_folder, exist_ok=True)

        nrrd_path = f"{case_folder}/{uploaded_nrrd.name}"

        with open(nrrd_path, "wb") as f:
            f.write(uploaded_nrrd.getbuffer())

        st.success("NRRD uploaded")

        if st.button("🚀 Run AI"):

            st.session_state.nrrd_path = nrrd_path
            seg_out = os.path.join(case_folder, "seg.seg.nrrd")

            with st.spinner("Running 3D AI Segmentation..."):
                st.session_state.seg_path = run_segmentation(nrrd_path, seg_out)

            with st.spinner("Calculating Sinus Metrics..."):
                st.session_state.left, st.session_state.right = run_metrics(st.session_state.seg_path)

            # ---------- RULE-BASED GENDER ----------
            total_volume = 0

            if st.session_state.left:
                total_volume += st.session_state.left[0]

            if st.session_state.right:
                total_volume += st.session_state.right[0]

            if total_volume > 0:
                st.session_state.gender = predict_gender_from_volume(total_volume)
            else:
                st.session_state.gender = None

            st.session_state.analysis_done = True
            st.success("Analysis Complete")


# ================= RESULTS =================
if st.session_state.analysis_done:

    nrrd_path = st.session_state.nrrd_path
    seg_path = st.session_state.seg_path
    left = st.session_state.left
    right = st.session_state.right

    st.header("📊 Measurements")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("LEFT SINUS")
        st.write(left)

    with col2:
        st.subheader("RIGHT SINUS")
        st.write(right)

    st.header("🧬 Gender Prediction")

    if st.session_state.gender:
        st.success(f"Predicted Gender: {st.session_state.gender}")
    else:
        st.warning("Gender could not be predicted.")

    st.header("🧠 Visualization")
    show_slices(nrrd_path, seg_path)

    st.header("💾 Saving Patient Report")

    patient_folder = os.path.join(
        "results",
        os.path.basename(nrrd_path).split('.')[0]
    )

    os.makedirs("results", exist_ok=True)
    save_medical_volumes(nrrd_path, seg_path, patient_folder)
    save_preview_image(nrrd_path, seg_path, patient_folder)

    excel_path = save_excel(
        left,
        right,
        st.session_state.gender,
        patient_folder
    )

    with open(excel_path, "rb") as f:
        st.download_button("📥 Download Excel Report", f, file_name="measurements.xlsx")