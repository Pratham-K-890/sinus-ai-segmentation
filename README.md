# 🦷 CBCT Sinus AI Analyzer

A Streamlit web app that takes CBCT (Cone Beam CT) scans of the maxillofacial region, segments the maxillary sinuses using a 3D AI model, computes sinus volume/metrics, and predicts gender based on sinus volume — a technique used in forensic and anthropological research.

## Features

- **Two input modes:** upload a raw DICOM folder or a pre-converted NRRD volume
- **DICOM → NRRD conversion** for raw scan folders
- **3D AI segmentation** of left and right maxillary sinuses
- **Sinus metrics** (volume, etc.) computed per side
- **Rule-based gender prediction** from total sinus volume
- **Interactive slice visualization** of the scan and segmentation
- **Exportable patient report**: NRRD volumes, preview image, and an Excel (`.xlsx`) report with measurements and predicted gender

## Project Structure

```
.
├── app.py                     # Streamlit application entry point
├── core/
│   ├── dicom_to_nrrd.py       # DICOM series → NRRD conversion
│   ├── segment.py             # 3D AI sinus segmentation
│   ├── metrics.py             # Left/right sinus measurement calculations
│   ├── visualize.py           # Slice-by-slice visualization
│   ├── export.py              # Save volumes, preview image, Excel report
│   └── gender_prediction.py   # Rule-based gender prediction from volume
├── requirements.txt
└── .gitignore
```

## Installation

```bash
git clone https://github.com/Pratham-K-890/sinus-ai-segmentation.git
cd sinus-ai-segmentation
pip install -r requirements.txt
```

**Dependencies:** `streamlit`, `SimpleITK`, `numpy`, `scipy`, `tensorflow`, `pynrrd`, `matplotlib`

## Usage

```bash
streamlit run app.py
```

1. Choose an input type — **Upload DICOM Folder** or **Upload NRRD File**.
2. Upload your scan data.
3. Click **🚀 Run AI** to run segmentation, metric calculation, and gender prediction.
4. Review the left/right sinus measurements, predicted gender, and slice visualizations.
5. Download the generated Excel report from the **💾 Saving Patient Report** section.

## Output

Results for each case are saved under `results/<case_name>/`, including the segmented volumes, a preview image, and an Excel measurements report.

## Disclaimer

This tool is intended for research and educational purposes only. It is **not a certified diagnostic or forensic tool** and should not be used for clinical or legal decision-making.

## License

Add a license of your choice (e.g., MIT) if you intend to share or accept contributions to this project.
