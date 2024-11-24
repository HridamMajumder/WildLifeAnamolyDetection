import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import pandas as pd
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Animal Species & Behavior Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths to example videos and result images
EXAMPLE_VIDEOS = {
    'original': r'D:\Applied Data Science and Aritifcial Intelligence\Zebra Tries to Kill Foal While Mother Fights Back - Latest Sightings (1080p, h264, youtube) (online-video-cutter.com).mp4',  # Replace with your video path
    'output': r'D:\Applied Data Science and Aritifcial Intelligence\output.mp4'       # Replace with your video path
}

RESULT_IMAGES = [
    {
        'path': 'path/to/result1.jpg',              # Replace with your image path
        'description': 'Description of result 1'
    },
    {
        'path': 'path/to/result2.jpg',              # Replace with your image path
        'description': 'Description of result 2'
    },
    {
        'path': 'path/to/result3.jpg',              # Replace with your image path
        'description': 'Description of result 3'
    },
    {
        'path': 'path/to/result4.jpg',              # Replace with your image path
        'description': 'Description of result 4'
    }
]

# Custom CSS for improved styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-header {
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .upload-section {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .results-section {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #34495e;
    }
    .video-container {
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None

def load_model(model_path):
    """Load YOLO model"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Sidebar Configuration
with st.sidebar:
    st.title("Configuration")
    
    # Model Selection Section
    st.header("Model Selection")
    
    # Model type selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Pre-trained Model", "Custom Model"]
    )
    
    if model_type == "Pre-trained Model":
        model_name = st.selectbox(
            "Select Pre-trained Model",
            ["YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"]
        )
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                model_path = f"{model_name.lower()}.pt"
                st.session_state.model = load_model(model_path)
                if st.session_state.model is not None:
                    st.success("Model loaded successfully!")
    
    else:
        model_file = st.file_uploader("Upload Custom Model", type=['pt'])
        if model_file is not None:
            if st.button("Load Custom Model"):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_model_path = os.path.join(temp_dir, "model.pt")
                        with open(temp_model_path, "wb") as f:
                            f.write(model_file.getvalue())
                        with st.spinner("Loading custom model..."):
                            st.session_state.model = load_model(temp_model_path)
                            if st.session_state.model is not None:
                                st.success("Custom model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    
    # Navigation
    st.header("Navigation")
    selected_tab = st.radio("Select Tab", ["Model Demo", "Example Output", "Report"])

# Main content
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("Animal Species & Behavior Detection")
st.markdown('</div>', unsafe_allow_html=True)

if selected_tab == "Model Demo":
    if st.session_state.model is None:
        st.warning("Please load a model in the sidebar first")
    else:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Image/Video", type=['jpg', 'jpeg', 'png', 'mp4'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            # Display original content
            st.subheader("Original Input")
            if uploaded_file.type.startswith('image'):
                st.image(uploaded_file, use_container_width=True)
            else:
                st.video(uploaded_file)
            
            # Display processing message
            st.info("Note: Real-time processing may take time. You can view example outputs in the Example Output tab.")
            
            # Optional: Process the file if needed
            if st.button("Process Input"):
                with st.spinner("Processing..."):
                    # Add your processing logic here
                    st.success("Processing complete!")

elif selected_tab == "Example Output":
    st.title("Example Model Output")
    
    # Create two columns for videos
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Video")
        try:
            st.video(EXAMPLE_VIDEOS['original'])
        except Exception as e:
            st.error(f"Error loading original video: Please check the video path")
    
    with col2:
        st.header("Model Output")
        try:
            st.video(EXAMPLE_VIDEOS['output'])
        except Exception as e:
            st.error(f"Error loading output video: Please check the video path")
    
    # Additional information
    st.markdown("""
    <div class="results-section">
        <h3>About this Example</h3>
        <p>This example demonstrates our model's capability in detecting and analyzing animal behavior. 
        The output video shows detected species and their behaviors in real-time.</p>
    </div>
    """, unsafe_allow_html=True)

elif selected_tab == "Report":
    st.title("Analysis Report")
    
    # Create grid for result images
    cols = st.columns(2)
    for idx, result in enumerate(RESULT_IMAGES):
        with cols[idx % 2]:
            try:
                st.image(result['path'], 
                        caption=f"Analysis Result {idx + 1}",
                        use_container_width=True)
                st.markdown(f"**Description:** {result['description']}")
            except Exception as e:
                st.error(f"Error loading image {idx + 1}: Please check the image path")
    
    # Summary section
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.subheader("Analysis Summary")
    
    # Summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Images Analyzed", len(RESULT_IMAGES))
    with col2:
        st.metric("Analysis Date", datetime.now().strftime("%Y-%m-%d"))
    
    # Additional summary information
    st.markdown("""
        ### Key Findings
        - Detection accuracy metrics
        - Behavior analysis results
        - Any anomalies detected
        
        ### Recommendations
        - Suggested improvements
        - Areas for further analysis
    """)
    st.markdown('</div>', unsafe_allow_html=True)
