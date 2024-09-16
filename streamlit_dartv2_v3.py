import math
import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
import os
import pandas as pd
import io
import time
from statistics import mean, stdev

# Parameters
RESULTS_PER_PAGE = 5
IMAGE_SIZE = 120  # Size of the image container in pixels
IMAGE_COLUMN_WIDTH = 1  # Width of the image column in the Streamlit grid
PREDICTION_COLUMN_WIDTH = 1  # Width of the prediction column in the Streamlit grid

# Set page configuration
st.set_page_config(page_title="DARTv2 - Fundus Image Analysis", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .main .block-container { padding-top: 2rem; }
    h1 { color: #2c3e50; }
    .stButton>button { background-color: #3498db; color: white; }
    .stDownloadButton>button { width: auto !important; }
    .image-container { width: 100%; display: flex; justify-content: center; align-items: center; }
    .image-container img { max-width: 100%; max-height: 100%; object-fit: contain; }
    .download-box { display: flex; align-items: center; gap: 10px; }
    .download-box > div { flex: 0 0 auto; }
    .format-select { width: 100px !important; }
    .result-card { background-color: white; padding: 0.5rem; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 0.5rem; }
    .metric-label { font-weight: bold; margin-bottom: 0.1rem; }
    .metric-value { font-size: 1.1rem; margin-bottom: 0.5rem; }
    .stHorizontalBlock { gap: 1rem !important; }
    .row-widget.stButton { text-align: left; }
    .result-text { display: flex; flex-direction: column; justify-content: center; height: 100%; }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
    .stMarkdown { margin-bottom: 0 !important; }
    .stImage { margin-bottom: 0.5rem; }

    /* Pagination styles */
    .pagination {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1rem;
    }
    .pagination button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        margin: 0 0.5rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .pagination button:disabled {
        background-color: #bdc3c7;
        cursor: not-allowed;
    }
    .pagination .page-number {
        font-weight: bold;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.onnx')
    return ort.InferenceSession(model_path)


def preprocess_image(image):
    image = image.convert('RGB').resize((256, 256))
    input_tensor = (np.asarray(image) / 255.0 - 0.5) / 0.5
    return np.expand_dims(np.transpose(input_tensor, (2, 0, 1)).astype('float32'), axis=0)


def predict(image, ort_session):
    input_tensor = preprocess_image(image)
    return ort_session.run(None, {'input': input_tensor})[0][0]


def create_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    return output.getvalue()


def create_json(df):
    return df.to_json(orient='records')


def process_new_files(new_files, ort_session):
    results = []
    processing_times = []
    total_files = len(new_files)
    progress_bar = st.progress(0)
    for idx, uploaded_file in enumerate(new_files):
        image = Image.open(uploaded_file)
        start_time = time.time()
        fractal_dimension, vessel_density = predict(image, ort_session)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)

        results.append({
            "File Name": uploaded_file.name,
            "Fractal_Dimension": fractal_dimension,
            "Vessel Density": vessel_density
        })
        progress_bar.progress(int(((idx+1)/total_files)*100))
    progress_bar.empty()
    return results, processing_times


def display_result(result, image):
    with st.container():
        cols = st.columns([1, 1])  # Equal width columns

        with cols[0]:  # Left column for image
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(Image.open(image), use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption(result["File Name"])

        with cols[1]:  # Right column for prediction text
            st.markdown('<div class="result-text">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Fractal Dimension</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{result["Fractal_Dimension"]:.4f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Vessel Density</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{result["Vessel Density"]:.4f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


def display_results(results, uploaded_images, start_idx, end_idx):
    for idx in range(start_idx, end_idx):
        display_result(results[idx], uploaded_images[idx])


def display_download_options(df_results):
    base_file_name = "DARTv2_results"
    st.markdown('<div class="download-box">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        download_format = st.selectbox("Format:", ["CSV", "Excel", "JSON"], key="download_format",
                                       label_visibility='collapsed')

    with col2:
        if download_format == "CSV":
            download_data = df_results.to_csv(index=False)
            file_name = base_file_name + ".csv"
            mime = "text/csv"
        elif download_format == "Excel":
            download_data = create_excel(df_results)
            file_name = base_file_name + ".xlsx"
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            download_data = create_json(df_results)
            file_name = base_file_name + ".json"
            mime = "application/json"

        st.download_button(
            label="Download Results",
            data=download_data,
            file_name=file_name,
            mime=mime,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def set_page(page_number):
    st.session_state.current_page = page_number


def create_pagination(total_pages, key_prefix):
    if total_pages > 1:
        prev_disabled = st.session_state.current_page == 1
        next_disabled = st.session_state.current_page == total_pages

        st.markdown('<div class="pagination">', unsafe_allow_html=True)
        cols = st.columns([1, 1, 1, 1, 1])
        with cols[0]:
            st.button('First', disabled=prev_disabled, key=f'{key_prefix}_first_page', on_click=set_page, args=(1,))
        with cols[1]:
            st.button('Previous', disabled=prev_disabled, key=f'{key_prefix}_prev_page', on_click=set_page, args=(st.session_state.current_page - 1,))
        with cols[2]:
            st.markdown(f'<div class="page-number">Page {st.session_state.current_page} of {total_pages}</div>', unsafe_allow_html=True)
        with cols[3]:
            st.button('Next', disabled=next_disabled, key=f'{key_prefix}_next_page', on_click=set_page, args=(st.session_state.current_page + 1,))
        with cols[4]:
            st.button('Last', disabled=next_disabled, key=f'{key_prefix}_last_page', on_click=set_page, args=(total_pages,))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.session_state.current_page = 1


def main():
    st.title("🔬 DARTv2 - Fundus Image Analysis")

    st.markdown("""
    This app uses the DARTv2 model to analyze fundus images and predict Fractal Dimension and Vessel Density.
    Upload one or more fundus images to get started!
    """)

    ort_session = load_model()

    uploaded_files = st.file_uploader("Choose fundus image(s)...", type=["jpg", "jpeg", "png", "bmp"],
                                      accept_multiple_files=True)

    if 'results' not in st.session_state:
        st.session_state.results = []
        st.session_state.uploaded_images = []
        st.session_state.processing_times = []
        st.session_state.current_page = 1  # Initialize current_page here

    if uploaded_files:
        st.session_state.results = [result for result in st.session_state.results if
                                    result["File Name"] in [file.name for file in uploaded_files]]
        st.session_state.uploaded_images = [image for image in st.session_state.uploaded_images if
                                            image.name in [file.name for file in uploaded_files]]

        new_files = [file for file in uploaded_files if file not in st.session_state.uploaded_images]
        if new_files:
            new_results, new_processing_times = process_new_files(new_files, ort_session)
            st.session_state.results.extend(new_results)
            st.session_state.uploaded_images.extend(new_files)
            st.session_state.processing_times.extend(new_processing_times)

        # Display Average Processing Time
        if st.session_state.processing_times:
            avg_time = mean(st.session_state.processing_times)
            std_time = stdev(st.session_state.processing_times) if len(st.session_state.processing_times) > 1 else 0
            # Display without arrow
            st.write(f"**Average Processing Time per Image (s):** {avg_time:.4f} ± {std_time:.4f}")

        st.write("### Uploaded Images and Results")
        total_pages = math.ceil(len(st.session_state.results) / RESULTS_PER_PAGE)

        # Pagination controls at the top
        create_pagination(total_pages, key_prefix='top')

        # Display results
        start_idx = (st.session_state.current_page - 1) * RESULTS_PER_PAGE
        end_idx = min(start_idx + RESULTS_PER_PAGE, len(st.session_state.results))
        display_results(st.session_state.results, st.session_state.uploaded_images, start_idx, end_idx)

        # Pagination controls at the bottom
        create_pagination(total_pages, key_prefix='bottom')

        df_results = pd.DataFrame(st.session_state.results)
        st.write("### Results Summary")
        st.dataframe(df_results)

        display_download_options(df_results)

        st.write("### Understanding the Results")
        st.write("""
        - **Fractal Dimension**: Quantifies the complexity of the retinal vasculature. Higher values (closer to 2.0) indicate a more complex branching pattern, generally associated with healthier retinal vessels.

        - **Vessel Density**: Represents the percentage of retinal area occupied by blood vessels. Higher values (closer to 1.0) often correlate with better retinal perfusion.

        These metrics provide insights into retinal health and may assist in early detection or monitoring of various ocular conditions. Interpret results with other clinical findings under professional guidance.
        """)

        st.write("### Citation")
        st.write("""
        If you use this tool in your research, please cite our work as follows:
        Pending publication. Please check back for updates on the final citation format.
        """)

    else:
        st.info("Please upload one or more fundus images to begin the analysis.")

    st.markdown("---")
    st.markdown("Created by Justin and Lucas")


if __name__ == "__main__":
    main()
