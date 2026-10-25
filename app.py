import streamlit as st
from PIL import Image
from src.predict import predict_disease, get_confidence_level, get_disease_info


# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Tomato Crop Disease Detector",
        page_icon="🍅",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def apply_custom_css():
    """Apply professional custom CSS styling."""
    st.markdown("""
    <style>
    /* Root variables */
    :root {
        --success-color: #2e7d32;
        --warning-color: #f57c00;
        --neutral-color: #0277bd;
        --light-bg: #f5f5f5;
        --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
    }
    
    /* Main container */
    .main {
        background-color: #fafafa;
    }
    
    /* Header hero section */
    .header-hero {
        background: #ffffff;
        padding: 2.5rem 2rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border-bottom: 3px solid #2e7d32;
    }
    
    .header-hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        color: #1b5e20;
        letter-spacing: -0.3px;
    }
    
    .header-hero h2 {
        font-size: 1.1rem;
        font-weight: 500;
        margin: 0 0 0.75rem 0;
        color: #558b2f;
    }
    
    .header-hero p {
        font-size: 0.95rem;
        margin: 0;
        color: #666;
        max-width: 650px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.5;
    }
    
    /* Diagnosis card */
    .diagnosis-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 2rem;
        box-shadow: var(--card-shadow);
        margin-bottom: 1.5rem;
    }
    
    .diagnosis-card.healthy {
        border-top: 4px solid #2e7d32;
    }
    
    .diagnosis-card.disease-high {
        border-top: 4px solid #d32f2f;
    }
    
    .diagnosis-card.disease-moderate {
        border-top: 4px solid #f57c00;
    }
    
    .diagnosis-card.disease-low {
        border-top: 4px solid #0277bd;
    }
    
    .diagnosis-label {
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .diagnosis-disease {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 1.2rem;
    }
    
    .diagnosis-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
    }
    
    .diagnosis-metric {
        text-align: center;
    }
    
    .diagnosis-metric-label {
        font-size: 0.75rem;
        color: #999;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }
    
    .diagnosis-metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    
    .healthy-value {
        color: #2e7d32;
    }
    
    .disease-value {
        color: #d32f2f;
    }
    
    .moderate-value {
        color: #f57c00;
    }
    
    .low-value {
        color: #0277bd;
    }
    
    /* Empty state */
    .empty-state {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 3rem 2rem;
        text-align: center;
        color: #888;
        box-shadow: var(--card-shadow);
    }
    
    .empty-state h3 {
        color: #555;
        margin-top: 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .empty-state p {
        color: #999;
        margin: 0.5rem 0 0 0;
    }
    
    /* Prediction section */
    .predictions-section {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: var(--card-shadow);
        margin-bottom: 1.5rem;
    }
    
    .prediction-item {
        padding: 0.75rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .prediction-item:last-child {
        border-bottom: none;
    }
    
    .prediction-name {
        font-weight: 500;
        color: #333;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    
    .prediction-conf {
        font-size: 0.8rem;
        color: #888;
    }
    
    /* Disease info cards */
    .disease-info-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: var(--card-shadow);
        border-left: 4px solid #2e7d32;
    }
    
    .disease-info-title {
        font-weight: 700;
        font-size: 0.85rem;
        color: #2e7d32;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
    }
    
    .disease-info-content {
        font-size: 0.9rem;
        color: #555;
        line-height: 1.6;
    }
    
    /* Healthy card */
    .healthy-card {
        background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%);
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        box-shadow: var(--card-shadow);
        border: 1px solid #c8e6c9;
    }
    
    .healthy-emoji {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .healthy-title {
        color: #2e7d32;
        font-weight: 700;
        font-size: 1.2rem;
        margin: 0 0 0.5rem 0;
    }
    
    .healthy-text {
        color: #558b2f;
        margin: 0;
        font-size: 0.95rem;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background-color: #e0e0e0;
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Sidebar styles */
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    
    .sidebar-section h3 {
        font-size: 0.9rem;
        font-weight: 600;
        color: #2e7d32;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    .sidebar-list {
        font-size: 0.85rem;
        color: #666;
        line-height: 1.7;
    }
    
    /* Warning messages */
    .warning-low-confidence {
        background-color: #fff3e0;
        border-left: 4px solid #f57c00;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
    }
    
    .warning-low-confidence p {
        margin: 0;
        color: #e65100;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    </style>
    """, unsafe_allow_html=True)




# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None


# ============================================================================
# HEADER SECTION
# ============================================================================

def render_header():
    """Render professional header hero section."""
    st.markdown("""
    <div class='header-hero'>
        <h1>🍅 Tomato Crop Disease Detector</h1>
        <h2>AI-Powered Disease Diagnosis & Crop Health</h2>
        <p>Upload a clear tomato leaf image to detect disease symptoms and receive treatment recommendations.</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar with information sections."""
    with st.sidebar:
        st.markdown("### About This Tool", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='sidebar-section'>
            <h3>AI Diagnosis System</h3>
            <div class='sidebar-list'>
                • Analyzes tomato leaf images<br>
                • Detects 10 disease conditions<br>
                • Provides top-3 predictions<br>
                • Confidence-based validation
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='sidebar-section'>
            <h3>Supported Conditions</h3>
            <div class='sidebar-list'>
                • Healthy<br>
                • Bacterial Spot<br>
                • Early Blight<br>
                • Late Blight<br>
                • Leaf Mold<br>
                • Septoria Leaf Spot<br>
                • Spider Mites<br>
                • Target Spot<br>
                • Yellow Leaf Curl Virus<br>
                • Tomato Mosaic Virus
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='sidebar-section'>
            <h3>Best Image Tips</h3>
            <div class='sidebar-list'>
                • Clear, close-up leaf images<br>
                • Natural, bright lighting<br>
                • Single leaf preferred<br>
                • Avoid blurry photos<br>
                • Avoid screenshots<br>
                • Avoid non-leaf objects
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='sidebar-section'>
            <h3>Disclaimer</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning(
            "Educational demonstration only. Not a substitute for professional agricultural expertise."
        )


# ============================================================================
# LEFT COLUMN - IMAGE UPLOAD
# ============================================================================

def render_upload_section(col):
    """Render image upload section in left column."""
    with col:
        st.subheader("Upload Tomato Leaf Image")
        
        uploaded_file = st.file_uploader(
            "Select image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = image
                st.image(image, caption="Uploaded Image", width="stretch")
            except Exception:
                st.error("Could not open image. Please upload a valid file.")
                st.session_state.uploaded_image = None
        else:
            st.markdown("""
            <div class='empty-state'>
                <h3>No Image Selected</h3>
                <p>Upload a tomato leaf image to begin</p>
            </div>
            """, unsafe_allow_html=True)
        
        return uploaded_file


def render_analyze_button():
    """Render analyze button."""
    return st.button(
        "Analyze Disease",
        use_container_width=True,
        type="primary",
        disabled=(st.session_state.uploaded_image is None)
    )


# ============================================================================
# RIGHT COLUMN - RESULTS
# ============================================================================

def get_diagnosis_card_class(disease_name, confidence):
    """Determine CSS class for diagnosis card based on disease and confidence."""
    if disease_name == "Healthy":
        return "metric-card healthy"
    
    if confidence >= 85:
        return "metric-card high-confidence"
    elif confidence >= 60:
        return "metric-card moderate-confidence"
    else:
        return "metric-card low-confidence"


def render_empty_results(col):
    """Render empty state in right column."""
    with col:
        st.subheader("Diagnosis Results")
        st.markdown("""
        <div class='empty-state'>
            <h3>No Analysis Yet</h3>
            <p>Upload an image and click Analyze to begin</p>
        </div>
        """, unsafe_allow_html=True)


def render_primary_diagnosis(col, result):
    """Render primary diagnosis card."""
    with col:
        st.subheader("Diagnosis Results")

        disease_name = result["disease_name"]
        confidence = result["confidence"]
        confidence_level = result["confidence_level"]

        # Healthy special case
        if disease_name == "Healthy":
            st.success("🌱 Plant Appears Healthy")
            st.info("No visible disease symptoms detected. Continue regular crop monitoring.")

        # Disease case
        else:
            st.error(f"🩺 Detected Condition: {disease_name}")

        # Metrics row
        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                label="Confidence",
                value=f"{confidence:.1f}%"
            )

        with metric_col2:
            st.metric(
                label="Confidence Level",
                value=confidence_level
            )

        # Low confidence warning
        if confidence < 75:
            st.warning(
                "⚠ Prediction confidence is low. Symptoms may overlap across diseases. "
                "Try uploading a clearer close-up tomato leaf image."
            )

def render_alternative_predictions(col, predictions):
    """Render top 3 predictions in right column."""
    with col:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.subheader("All Predictions")
        
        for idx, pred in enumerate(predictions[:3], 1):
            label = pred["label"]
            conf = pred["confidence"]
            
            st.markdown(f"""
            <div class='prediction-row'>
                <div class='prediction-label'>{idx}. {label}</div>
                <div class='prediction-confidence'>{conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(min(conf / 100, 1.0))


def render_low_confidence_warning(col):
    """Render low confidence warning."""
    with col:
        st.warning(
            "Prediction confidence is low. Image may be unclear or symptoms may overlap. "
            "Try uploading a clearer close-up image."
        )


def render_healthy_card(col):
    """Render special healthy condition card."""
    with col:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #e8f5e9; border-radius: 8px; padding: 2rem; text-align: center;'>
            <div style='font-size: 2.5rem; margin-bottom: 1rem;'>🌱</div>
            <h3 style='color: #388e3c; margin: 0 0 0.5rem 0;'>Plant Appears Healthy</h3>
            <p style='color: #555; margin: 0;'>No visible disease symptoms detected. Continue regular crop monitoring.</p>
        </div>
        """, unsafe_allow_html=True)


def render_disease_information(col, disease_info):
    """Render disease information in two-column layout."""
    with col:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.subheader("Disease Information")
        
        # Two column layout for disease details
        info_col1, info_col2 = st.columns(2)
        
        # Left column - Symptoms and Prevention
        with info_col1:
            if disease_info.get("symptoms"):
                st.markdown(f"""
                <div class='disease-info-column'>
                    <div class='disease-info-title'>Symptoms</div>
                    <div class='disease-info-content'>{disease_info['symptoms']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if disease_info.get("prevention"):
                st.markdown(f"""
                <div class='disease-info-column'>
                    <div class='disease-info-title'>Prevention</div>
                    <div class='disease-info-content'>{disease_info['prevention']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Right column - Description and Treatment
        with info_col2:
            if disease_info.get("description"):
                st.markdown(f"""
                <div class='disease-info-column'>
                    <div class='disease-info-title'>Description</div>
                    <div class='disease-info-content'>{disease_info['description']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if disease_info.get("treatment"):
                st.markdown(f"""
                <div class='disease-info-column'>
                    <div class='disease-info-title'>Treatment</div>
                    <div class='disease-info-content'>{disease_info['treatment']}</div>
                </div>
                """, unsafe_allow_html=True)


# ============================================================================
# PREDICTION HANDLER
# ============================================================================

def handle_prediction(image):
    """Execute prediction with spinner and error handling."""
    with st.spinner("Analyzing tomato leaf..."):
        try:
            result = predict_disease(image)
            
            # Handle invalid/rejected images
            if not result["valid"]:
                st.session_state.prediction_result = {
                    "valid": False,
                    "error": result["error"]
                }
                return
            
            predictions = result.get("predictions", [])
            if not predictions:
                st.session_state.prediction_result = {
                    "valid": False,
                    "error": "No predictions generated. Please try another image."
                }
                return
            
            # Extract primary prediction
            primary = predictions[0]
            disease_name = primary.get("label", "Unknown")
            confidence = primary.get("confidence", 0)
            
            # Get confidence level and disease info
            confidence_level = get_confidence_level(confidence)
            disease_info = get_disease_info(disease_name)
            
            # Store result in session state
            st.session_state.prediction_result = {
                "valid": True,
                "predictions": predictions,
                "disease_name": disease_name,
                "confidence": confidence,
                "confidence_level": confidence_level,
                "disease_info": disease_info
            }
        
        except Exception:
            st.session_state.prediction_result = {
                "valid": False,
                "error": "Analysis failed. Please try another image."
            }


# ============================================================================
# FOOTER
# ============================================================================

def render_footer():
    """Render footer."""
    st.markdown("""
    <div class='footer'>
        <p>Built with Streamlit + TensorFlow | AI Tomato Disease Detection Project</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application flow."""
    configure_page()
    apply_custom_css()
    init_session_state()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Main dashboard - two column layout
    left_col, right_col = st.columns([1, 1], gap="large")
    
    # LEFT COLUMN - Upload
    uploaded_file = render_upload_section(left_col)
    
    with left_col:
        st.write("")
        if render_analyze_button():
            if st.session_state.uploaded_image is not None:
                handle_prediction(st.session_state.uploaded_image)
    
    # RIGHT COLUMN - Results
    result = st.session_state.prediction_result
    
    if result is None:
        render_empty_results(right_col)
    elif not result.get("valid"):
        with right_col:
            st.subheader("Diagnosis Results")
            st.error(f"❌ {result.get('error', 'Analysis failed.')}")
    else:
        render_primary_diagnosis(right_col, result)
        render_alternative_predictions(right_col, result["predictions"])
        
        if result["confidence"] < 60:
            render_low_confidence_warning(right_col)
        
        if result["disease_name"] == "Healthy":
            render_healthy_card(right_col)
        else:
            render_disease_information(right_col, result["disease_info"])
    
    # Footer
    st.write("")
    render_footer()


if __name__ == "__main__":
    main()
