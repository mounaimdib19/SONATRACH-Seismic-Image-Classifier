import streamlit as st
import cv2
import numpy as np
import os
import zipfile
import shutil
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="SONATRACH Seismic Classifier",
    page_icon="🔍",
    layout="wide"
)

class SeismicClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, image):
        """Extract features from seismic image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        gray = gray.astype(np.float32) / 255.0
        
        features = []
        
        # Statistical features
        features.extend([
            np.mean(gray), np.std(gray), np.var(gray),
            np.min(gray), np.max(gray)
        ])
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(gradient_mag), np.std(gradient_mag)
        ])
        
        # Frequency features
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        features.extend([
            np.mean(magnitude), np.std(magnitude)
        ])
        
        # Noise features
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blur
        
        features.extend([
            np.std(noise), np.mean(np.abs(noise))
        ])
        
        return np.array(features)
    
    def train(self, clean_images, noisy_images):
        """Train the classifier"""
        X = []
        y = []
        
        for img in clean_images:
            features = self.extract_features(img)
            X.append(features)
            y.append(0)
        
        for img in noisy_images:
            features = self.extract_features(img)
            X.append(features)
            y.append(1)
        
        X = np.array(X)
        y = np.array(y)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
        
        return len(clean_images), len(noisy_images)
    
    def predict(self, image):
        """Predict if image is clean or noisy"""
        if not self.is_trained:
            return None
        
        features = self.extract_features(image).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'is_noisy': bool(prediction),
            'confidence': float(max(probability)),
            'clean_prob': float(probability[0]),
            'noisy_prob': float(probability[1])
        }

def initialize_session_state():
    """Initialize session state variables"""
    if 'classifier' not in st.session_state:
        st.session_state.classifier = SeismicClassifier()
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'test_images' not in st.session_state:
        st.session_state.test_images = []
    if 'test_filenames' not in st.session_state:
        st.session_state.test_filenames = []
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'clean_folder' not in st.session_state:
        st.session_state.clean_folder = 'output_clean'
    if 'noisy_folder' not in st.session_state:
        st.session_state.noisy_folder = 'output_noisy'

def load_image_from_uploaded_file(uploaded_file):
    """Load image from uploaded file"""
    try:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        if len(image_array.shape) == 2:  # Grayscale
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        return image_array
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def create_zip_file(folder_path, zip_name):
    """Create a zip file from a folder"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zip_file.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer

def save_results_to_folders(results, test_images, test_filenames, clean_folder, noisy_folder):
    """Save classified images to separate folders"""
    # Create folders if they don't exist
    os.makedirs(clean_folder, exist_ok=True)
    os.makedirs(noisy_folder, exist_ok=True)
    
    for result, image, filename in zip(results, test_images, test_filenames):
        if result['is_noisy']:
            output_path = os.path.join(noisy_folder, filename)
        else:
            output_path = os.path.join(clean_folder, filename)
        
        # Convert RGB to BGR for cv2
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)

def main():
    initialize_session_state()
    
    # Header
    st.title("🔍 SONATRACH Seismic Image Classifier")
    st.markdown("### Automatic Classification of Seismic Data: Clean vs Noisy")
    st.markdown("---")
    
    # Sidebar for training
    with st.sidebar:
        st.header("📚 Training Data")
        st.markdown("Upload training images to train the classifier")
        
        st.subheader("Clean Images (No Noise)")
        clean_files = st.file_uploader(
            "Upload clean seismic images",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True,
            key="clean_uploader"
        )
        
        st.subheader("Noisy Images (With Interference)")
        noisy_files = st.file_uploader(
            "Upload noisy seismic images",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True,
            key="noisy_uploader"
        )
        
        if st.button("🎯 Train Model", type="primary"):
            if clean_files and noisy_files:
                with st.spinner("Training classifier..."):
                    clean_images = []
                    noisy_images = []
                    
                    for file in clean_files:
                        img = load_image_from_uploaded_file(file)
                        if img is not None:
                            clean_images.append(img)
                    
                    for file in noisy_files:
                        img = load_image_from_uploaded_file(file)
                        if img is not None:
                            noisy_images.append(img)
                    
                    clean_count, noisy_count = st.session_state.classifier.train(
                        clean_images, noisy_images
                    )
                    
                    st.success(f"✅ Model trained successfully!")
                    st.info(f"Clean images: {clean_count}\nNoisy images: {noisy_count}")
            else:
                st.error("Please upload both clean and noisy images!")
        
        st.markdown("---")
        st.markdown(f"**Model Status:** {'✅ Trained' if st.session_state.classifier.is_trained else '❌ Not Trained'}")
    
    # Main content
    if not st.session_state.classifier.is_trained:
        st.warning("⚠️ Please train the model first using the sidebar!")
        st.info("👈 Upload clean and noisy training images, then click 'Train Model'")
        
        # Show instructions
        st.markdown("### 📖 Instructions:")
        st.markdown("""
        1. **Upload Training Data** (Sidebar):
           - Upload several clean seismic images (without interference)
           - Upload several noisy seismic images (with interference, like Figure 16)
           - Click "Train Model"
        
        2. **Upload Test Dataset**:
           - Once trained, upload your test images below
           - Click "Start Classification"
        
        3. **Review Results**:
           - View each image classification one by one
           - Use "Next" button to go through all images
        
        4. **Download Results**:
           - Download organized folders (clean/noisy)
        """)
    
    else:
        # Test dataset upload
        st.header("📂 Upload Test Dataset")
        test_files = st.file_uploader(
            "Upload seismic images to classify",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True,
            key="test_uploader"
        )
        
        if test_files and st.button("🚀 Start Classification", type="primary"):
            with st.spinner("Loading images..."):
                st.session_state.test_images = []
                st.session_state.test_filenames = []
                st.session_state.results = []
                st.session_state.current_index = 0
                st.session_state.processing_complete = False
                
                for file in test_files:
                    img = load_image_from_uploaded_file(file)
                    if img is not None:
                        st.session_state.test_images.append(img)
                        st.session_state.test_filenames.append(file.name)
                
                # Classify all images
                for img in st.session_state.test_images:
                    result = st.session_state.classifier.predict(img)
                    st.session_state.results.append(result)
                
                st.success(f"✅ Loaded {len(st.session_state.test_images)} images!")
        
        # Display current image and results
        if st.session_state.test_images and st.session_state.results:
            st.markdown("---")
            st.header("📊 Classification Results")
            
            # Progress
            total_images = len(st.session_state.test_images)
            current_idx = st.session_state.current_index
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.progress((current_idx + 1) / total_images)
                st.markdown(f"### Image {current_idx + 1} of {total_images}")
            
            # Display current image and result
            current_image = st.session_state.test_images[current_idx]
            current_filename = st.session_state.test_filenames[current_idx]
            current_result = st.session_state.results[current_idx]
            
            # Layout
            col_left, col_right = st.columns([1, 1])
            
            with col_left:
                st.image(current_image, caption=current_filename, use_container_width=True)
            
            with col_right:
                st.markdown("### Classification Result")
                
                if current_result['is_noisy']:
                    st.error("🔴 NOISY - Contains Interference")
                    st.markdown(f"**Confidence:** {current_result['confidence']:.1%}")
                else:
                    st.success("🟢 CLEAN - No Interference")
                    st.markdown(f"**Confidence:** {current_result['confidence']:.1%}")
                
                # Probability bars
                st.markdown("#### Probability Distribution")
                st.progress(current_result['clean_prob'])
                st.markdown(f"Clean: **{current_result['clean_prob']:.1%}**")
                
                st.progress(current_result['noisy_prob'])
                st.markdown(f"Noisy: **{current_result['noisy_prob']:.1%}**")
            
            # Navigation buttons
            st.markdown("---")
            col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
            
            with col_nav1:
                if st.button("⬅️ Previous", disabled=(current_idx == 0)):
                    st.session_state.current_index -= 1
                    st.rerun()
            
            with col_nav2:
                if current_idx < total_images - 1:
                    if st.button("Next ➡️", type="primary"):
                        st.session_state.current_index += 1
                        st.rerun()
                else:
                    if not st.session_state.processing_complete:
                        st.session_state.processing_complete = True
                        # Save results to folders
                        save_results_to_folders(
                            st.session_state.results,
                            st.session_state.test_images,
                            st.session_state.test_filenames,
                            st.session_state.clean_folder,
                            st.session_state.noisy_folder
                        )
                        st.rerun()
            
            with col_nav3:
                st.markdown(f"**{current_idx + 1} / {total_images}**")
            
            # Summary statistics after completion
            if st.session_state.processing_complete:
                st.markdown("---")
                st.header("📈 Summary")
                
                clean_count = sum(1 for r in st.session_state.results if not r['is_noisy'])
                noisy_count = sum(1 for r in st.session_state.results if r['is_noisy'])
                avg_confidence = np.mean([r['confidence'] for r in st.session_state.results])
                
                col_s1, col_s2, col_s3 = st.columns(3)
                
                with col_s1:
                    st.metric("Total Images", total_images)
                
                with col_s2:
                    st.metric("Clean Images", clean_count)
                
                with col_s3:
                    st.metric("Noisy Images", noisy_count)
                
                st.info(f"**Average Confidence:** {avg_confidence:.1%}")
                
                # Create detailed results table
                results_df = pd.DataFrame({
                    'Filename': st.session_state.test_filenames,
                    'Classification': ['Noisy' if r['is_noisy'] else 'Clean' 
                                     for r in st.session_state.results],
                    'Confidence': [f"{r['confidence']:.1%}" 
                                 for r in st.session_state.results],
                    'Clean Probability': [f"{r['clean_prob']:.1%}" 
                                        for r in st.session_state.results],
                    'Noisy Probability': [f"{r['noisy_prob']:.1%}" 
                                        for r in st.session_state.results]
                })
                
                st.markdown("### Detailed Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Download options
                st.markdown("---")
                st.header("📥 Download Results")
                
                col_d1, col_d2, col_d3 = st.columns(3)
                
                with col_d1:
                    if os.path.exists(st.session_state.clean_folder) and os.listdir(st.session_state.clean_folder):
                        clean_zip = create_zip_file(st.session_state.clean_folder, "clean_images.zip")
                        st.download_button(
                            label="📦 Download Clean Images",
                            data=clean_zip,
                            file_name=f"clean_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            type="primary"
                        )
                    else:
                        st.info("No clean images to download")
                
                with col_d2:
                    if os.path.exists(st.session_state.noisy_folder) and os.listdir(st.session_state.noisy_folder):
                        noisy_zip = create_zip_file(st.session_state.noisy_folder, "noisy_images.zip")
                        st.download_button(
                            label="📦 Download Noisy Images",
                            data=noisy_zip,
                            file_name=f"noisy_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            type="primary"
                        )
                    else:
                        st.info("No noisy images to download")
                
                with col_d3:
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV Report",
                        data=csv,
                        file_name=f"classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        type="primary"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>SONATRACH - Direction Centrale Recherche & Développement</strong></p>
        <p>Seismic Data Deblending Project - Automatic Classification System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()