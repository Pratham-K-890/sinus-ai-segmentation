import os
import SimpleITK as sitk


def convert_dicom_to_nrrd(input_folder, output_folder):

    if not os.path.exists(input_folder):
        print("❌ Input folder does not exist.")
        return None

    os.makedirs(output_folder, exist_ok=True)

    reader = sitk.ImageSeriesReader()

    # Detect DICOM series
    series_ids = reader.GetGDCMSeriesIDs(input_folder)

    if not series_ids:
        print("❌ No DICOM series found.")
        return None

    print(f"✅ Found {len(series_ids)} series")

    # Use first series (CBCT usually has one)
    series_id = series_ids[0]

    dicom_files = reader.GetGDCMSeriesFileNames(input_folder, series_id)

    reader.SetFileNames(dicom_files)

    print("⏳ Reading DICOM volume...")
    image = reader.Execute()

    # Use parent folder name
    patient_name = os.path.basename(os.path.abspath(input_folder))

    output_path = os.path.join(output_folder, f"{patient_name}.nrrd")

    print("⏳ Writing NRRD file...")
    sitk.WriteImage(image, output_path)

    print(f"✅ Saved: {output_path}")
    print("📏 Spacing:", image.GetSpacing())
    print("📐 Size:", image.GetSize())

    return output_path
