import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Adult Income Classifier",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">💰 Adult Income Classification System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict Income >50K or ≤50K using Census Data | ML Assignment 2</p>', unsafe_allow_html=True)
st.markdown("---")

# Load models function
@st.cache_resource
def load_models():
    """Load all trained models and preprocessors"""
    try:
        models = {
            'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
            'Decision Tree': joblib.load('models/decision_tree.pkl'),
            'K-Nearest Neighbors': joblib.load('models/knn.pkl'),
            'Naive Bayes': joblib.load('models/naive_bayes.pkl'),
            'Random Forest': joblib.load('models/random_forest.pkl'),
            'XGBoost': joblib.load('models/xgboost.pkl')
        }
        scaler = joblib.load('models/scaler.pkl')
        return models, scaler, None
    except Exception as e:
        return None, None, str(e)

# Evaluation function
def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    results = {}
    results['Accuracy'] = accuracy_score(y_true, y_pred)
    results['Precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    results['Recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    results['F1 Score'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    results['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    if y_pred_proba is not None:
        try:
            results['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except:
            results['AUC'] = 'N/A'
    else:
        results['AUC'] = 'N/A'
    
    return results

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Load models
with st.spinner('🔄 Loading models...'):
    models, scaler, error = load_models()

if models is None:
    st.sidebar.error("❌ Failed to load models")
    st.error(f"**Error loading models:** {error}")
    st.info("""
    **Possible causes:**
    1. Models folder not found
    2. Model files (.pkl) missing
    3. Incompatible scikit-learn version
    
    **Solution:**
    - Ensure `models/` folder exists with all 8 .pkl files
    - Check GitHub repository structure
    - Verify requirements.txt versions
    """)
    st.stop()
else:
    st.sidebar.success("✅ Models loaded successfully!")

# Model info
st.sidebar.markdown("### 📊 Available Models")
st.sidebar.markdown("""
- **Logistic Regression** (82.4%)
- **Decision Tree** (84.8%)
- **K-Nearest Neighbors** (83.0%)
- **Naive Bayes** (79.6%)
- **Random Forest** (86.7%)
- **XGBoost** (87.5%) ⭐
""")

# Model selection
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Select Model")
model_choice = st.sidebar.selectbox(
    "Choose a classification model:",
    list(models.keys()),
    index=5,  # Default to XGBoost
    help="Select which model to use for predictions"
)

# Display model details
model_info = {
    'Logistic Regression': {'accuracy': '82.4%', 'type': 'Linear', 'speed': 'Fast'},
    'Decision Tree': {'accuracy': '84.8%', 'type': 'Tree-based', 'speed': 'Fast'},
    'K-Nearest Neighbors': {'accuracy': '83.0%', 'type': 'Instance-based', 'speed': 'Slow'},
    'Naive Bayes': {'accuracy': '79.6%', 'type': 'Probabilistic', 'speed': 'Very Fast'},
    'Random Forest': {'accuracy': '86.7%', 'type': 'Ensemble', 'speed': 'Fast'},
    'XGBoost': {'accuracy': '87.5%', 'type': 'Ensemble', 'speed': 'Fast'}
}

info = model_info[model_choice]
st.sidebar.info(f"""
**Model:** {model_choice}  
**Type:** {info['type']}  
**Accuracy:** {info['accuracy']}  
**Speed:** {info['speed']}
""")

# File upload section
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Upload Test Data")

# Download sample data button
st.sidebar.markdown("**Need sample data?**")
try:
    import os
    if os.path.exists('test_data.csv'):
        with open('test_data.csv', 'r') as f:
            sample_data = f.read()
        st.sidebar.download_button(
            label="📥 Download Sample CSV",
            data=sample_data,
            file_name="sample_test_data.csv",
            mime="text/csv",
            help="Download sample test data to try the app",
            use_container_width=True
        )
    else:
        st.sidebar.info("💡 Sample test_data.csv not found. Upload your own CSV file below.")
except Exception as e:
    st.sidebar.info("💡 Upload your CSV file below to begin")

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload test dataset with 'target' column (0: ≤50K, 1: >50K)"
)

# File format info
with st.sidebar.expander("ℹ️ CSV Format Requirements"):
    st.markdown("""
    **Required columns:**
    - All 14 Adult Income features
    - `target` column (0 or 1)
    
    **Features:**
    - age, workclass, fnlwgt, education
    - education-num, marital-status, occupation
    - relationship, race, sex, capital-gain
    - capital-loss, hours-per-week, native-country
    
    **Note:** Categorical features should be encoded as integers
    """)

# About section
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About This App"):
    st.markdown("""
    **ML Assignment 2**  
    Adult Income Classification
    
    **Student:** TUSHAR KANTI SANTRA  
    **ID:** 2025AB05283  
    **Institution:** BITS Pilani
    
    **Dataset:** UCI Adult Income  
    48,842 samples | 14 features
    
    **GitHub:** [Repository Link](https://github.com/Tks008/ML_Classification_Streamlit_Assignment)
    """)

# Main content
if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Success metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Samples", f"{df.shape[0]:,}")
        with col2:
            st.metric("📈 Features", df.shape[1] - 1 if 'target' in df.columns else df.shape[1])
        with col3:
            if 'target' in df.columns:
                st.metric("✅ Target Column", "Present")
            else:
                st.metric("❌ Target Column", "Missing")
        with col4:
            st.metric("🤖 Model", model_choice)
        
        # Data preview
        with st.expander("📋 Data Preview (First 10 Rows)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Check for target column
        if 'target' not in df.columns:
            st.error("❌ **Error:** 'target' column not found in uploaded file!")
            st.info("💡 **Tip:** Please ensure your CSV has a 'target' column with values 0 (≤50K) or 1 (>50K)")
            st.stop()
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Display target distribution
        st.markdown("---")
        st.subheader("🎯 Target Distribution Analysis")
        
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            st.markdown("##### Class Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            target_counts = y.value_counts()
            colors = ['#FF6B6B', '#4ECDC4']
            bars = ax.bar(['≤50K', '>50K'], target_counts.values, color=colors, alpha=0.8, edgecolor='black')
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Target Class Distribution', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("##### Percentage Split")
            fig, ax = plt.subplots(figsize=(6, 4))
            percentages = y.value_counts(normalize=True) * 100
            ax.pie(target_counts.values, labels=['≤50K', '>50K'], 
                   autopct='%1.1f%%', colors=colors, startangle=90,
                   textprops={'fontsize': 11, 'fontweight': 'bold'})
            ax.set_title('Target Class Percentage', fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col3:
            st.markdown("##### Quick Statistics")
            total = len(y)
            class_0 = (y == 0).sum()
            class_1 = (y == 1).sum()
            
            st.markdown(f"""
            **Total Samples:** {total:,}
            
            **Class Breakdown:**
            - ≤50K (Class 0): {class_0:,} ({class_0/total*100:.1f}%)
            - >50K (Class 1): {class_1:,} ({class_1/total*100:.1f}%)
            
            **Class Imbalance Ratio:** {class_0/class_1:.2f}:1
            
            {'⚠️ **Dataset is imbalanced**' if class_0/class_1 > 2 else '✅ **Dataset is balanced**'}
            """)
        
        # Process features
        st.markdown("---")
        st.subheader(f"🔮 Predictions using {model_choice}")
        
        # Get selected model
        selected_model = models[model_choice]
        
        # Determine if scaling is needed
        models_need_scaling = ['Logistic Regression', 'K-Nearest Neighbors']
        
        with st.spinner('⚙️ Preprocessing features...'):
            if model_choice in models_need_scaling:
                X_processed = scaler.transform(X)
                st.info(f"✓ Features scaled using StandardScaler (required for {model_choice})")
            else:
                X_processed = X.values
                st.info(f"✓ Using original features (no scaling needed for {model_choice})")
        
        # Make predictions
        with st.spinner('🔄 Making predictions...'):
            y_pred = selected_model.predict(X_processed)
            y_pred_proba = selected_model.predict_proba(X_processed)
        
        st.success("✅ Predictions completed successfully!")
        
        # Calculate metrics
        results = evaluate_model(y, y_pred, y_pred_proba)
        
        # Display metrics in cards
        st.markdown("---")
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        metrics_data = [
            (col1, "Accuracy", results['Accuracy'], "📈"),
            (col2, "AUC Score", results['AUC'], "📉"),
            (col3, "Precision", results['Precision'], "🎯"),
            (col4, "Recall", results['Recall'], "🔍"),
            (col5, "F1 Score", results['F1 Score'], "⚖️"),
            (col6, "MCC", results['MCC'], "🧮")
        ]
        
        for col, name, value, icon in metrics_data:
            with col:
                if isinstance(value, str):
                    st.metric(f"{icon} {name}", value)
                else:
                    st.metric(f"{icon} {name}", f"{value:.4f}")
        
        # Confusion Matrix and ROC Curve
        st.markdown("---")
        st.markdown("### 📈 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(7, 6))
            cm = confusion_matrix(y, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Count'},
                       xticklabels=['≤50K', '>50K'],
                       yticklabels=['≤50K', '>50K'],
                       annot_kws={'fontsize': 14, 'fontweight': 'bold'})
            ax.set_title(f'{model_choice}\nConfusion Matrix', fontsize=13, fontweight='bold', pad=15)
            ax.set_ylabel('Actual Class', fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Confusion matrix interpretation
            st.markdown("""
            **Interpretation:**
            - **True Negatives (TN):** Correctly predicted ≤50K
            - **True Positives (TP):** Correctly predicted >50K
            - **False Positives (FP):** Incorrectly predicted >50K
            - **False Negatives (FN):** Incorrectly predicted ≤50K
            """)
        
        with col2:
            st.markdown("#### 📈 ROC Curve")
            if y_pred_proba is not None and results['AUC'] != 'N/A':
                fig, ax = plt.subplots(figsize=(7, 6))
                fpr, tpr, _ = roc_curve(y, y_pred_proba[:, 1])
                ax.plot(fpr, tpr, color='#4ECDC4', linewidth=3, 
                       label=f'ROC Curve (AUC = {results["AUC"]:.4f})')
                ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
                ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
                ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
                ax.set_title(f'{model_choice}\nROC Curve', fontsize=13, fontweight='bold', pad=15)
                ax.legend(loc='lower right', fontsize=10)
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("ROC Curve not available for this model configuration")
            
            # Additional metrics
            st.markdown("#### 📋 Quick Stats")
            total_samples = len(y)
            correct_pred = (y == y_pred).sum()
            incorrect_pred = total_samples - correct_pred
            
            st.markdown(f"""
            **Total Samples:** {total_samples:,}  
            **Correct Predictions:** {correct_pred:,} ({correct_pred/total_samples*100:.2f}%)  
            **Incorrect Predictions:** {incorrect_pred:,} ({incorrect_pred/total_samples*100:.2f}%)
            
            **Confusion Matrix Values:**
            - True Negatives: {cm[0,0]:,}
            - True Positives: {cm[1,1]:,}
            - False Positives: {cm[0,1]:,}
            - False Negatives: {cm[1,0]:,}
            """)
        
        # Classification Report
        st.markdown("---")
        st.markdown("### 📋 Detailed Classification Report")
        
        report_dict = classification_report(y, y_pred, 
                                           target_names=['≤50K (Class 0)', '>50K (Class 1)'],
                                           output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        
        # Style the dataframe
        st.dataframe(
            report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score'])
                           .format(precision=4),
            use_container_width=True
        )
        
        # Prediction Analysis
        st.markdown("---")
        st.markdown("### 📊 Prediction Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Predicted Distribution")
            pred_counts = pd.Series(y_pred).value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(['≤50K', '>50K'], 
                   [pred_counts.get(0, 0), pred_counts.get(1, 0)],
                   color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black')
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title('Predicted Classes', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Prediction Confidence")
            confidence = y_pred_proba.max(axis=1)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(confidence, bins=20, color='#45B7D1', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Confidence Score', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title('Prediction Confidence', fontsize=11, fontweight='bold')
            ax.axvline(confidence.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {confidence.mean():.3f}')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.metric("Average Confidence", f"{confidence.mean():.4f}")
        
        with col3:
            st.markdown("#### Sample Predictions")
            comparison_df = pd.DataFrame({
                'Actual': y.values[:15].astype(int),
                'Predicted': y_pred[:15].astype(int),
                'Confidence': [f"{conf:.3f}" for conf in confidence[:15]]
            })
            comparison_df['Actual'] = comparison_df['Actual'].map({0: '≤50K', 1: '>50K'})
            comparison_df['Predicted'] = comparison_df['Predicted'].map({0: '≤50K', 1: '>50K'})
            comparison_df['Match'] = ['✅' if comparison_df.loc[i, 'Actual'] == comparison_df.loc[i, 'Predicted'] else '❌' 
                                     for i in range(len(comparison_df))]
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Download predictions
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        
        results_df = df.copy()
        results_df['Predicted'] = y_pred
        results_df['Predicted_Class'] = results_df['Predicted'].map({0: '≤50K', 1: '>50K'})
        results_df['Confidence'] = y_pred_proba.max(axis=1)
        results_df['Correct'] = (y == y_pred).astype(int)
        
        csv = results_df.to_csv(index=False)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"predictions_{model_choice.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
    except Exception as e:
        st.error(f"❌ **Error processing file:** {str(e)}")
        with st.expander("🔍 View Error Details"):
            st.exception(e)
        
        st.info("""
        **Common Issues:**
        1. Missing 'target' column
        2. Incorrect feature names
        3. Wrong data types
        4. Missing features (need all 14)
        
        **Solution:**
        - Check CSV format matches training data
        - Ensure all features are present
        - Verify categorical features are encoded
        """)

else:
    # Instructions when no file uploaded
    st.info("👈 **Please upload a CSV file from the sidebar to begin predictions**")
    
    # Sample data download section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 📥 Quick Start")
        st.markdown("**Don't have test data? Download our sample file:**")
        
        try:
            import os
            if os.path.exists('test_data.csv'):
                with open('test_data.csv', 'r') as f:
                    sample_data = f.read()
                
                st.download_button(
                    label="⬇️ Download Sample Test Data (9,769 samples)",
                    data=sample_data,
                    file_name="sample_test_data.csv",
                    mime="text/csv",
                    help="Download sample Adult Income test data to try the app",
                    use_container_width=True,
                    type="primary"
                )
                
                st.caption("✅ This sample file contains 9,769 test samples from the Adult Income dataset with all features encoded and ready to use.")
            else:
                st.info("Sample data will be available after deployment. For now, prepare your own CSV with the format shown below.")
        except Exception as e:
            st.info("💡 Prepare your CSV file with the format shown below")
    
    st.markdown("---")
    
    # Two column layout for instructions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 How to Use This App")
        st.markdown("""
        1. **Select Model** from sidebar (default: XGBoost)
        2. **Upload CSV** file with test data
        3. **View Results** with comprehensive metrics
        4. **Analyze Performance** using visualizations
        5. **Download** predictions for further analysis
        
        ### ✨ App Features
        - ✅ **6 Classification Models** available
        - ✅ **Comprehensive Metrics** (Accuracy, AUC, Precision, Recall, F1, MCC)
        - ✅ **Visual Analytics** (Confusion Matrix, ROC Curve, Distribution Charts)
        - ✅ **Detailed Reports** (Classification Report, Prediction Analysis)
        - ✅ **Export Results** (Download predictions as CSV)
        - ✅ **Real-time Processing** (Instant predictions)
        """)
    
    with col2:
        st.markdown("### 📊 About the Dataset")
        st.markdown("""
        **Adult Income Dataset (UCI)**
        
        **Statistics:**
        - **Instances:** 48,842 samples
        - **Features:** 14 attributes
        - **Task:** Binary Classification
        - **Classes:** ≤50K (75%) vs >50K (25%)
        
        **Feature Categories:**
        - **Demographics:** age, sex, race
        - **Education:** education, education-num
        - **Work:** workclass, occupation, hours-per-week
        - **Financial:** capital-gain, capital-loss
        - **Other:** marital-status, relationship, native-country
        
        **Goal:** Predict if annual income exceeds $50,000
        """)
        
        # Sample data format
        st.markdown("### 📋 Sample CSV Format")
        sample_df = pd.DataFrame({
            'age': [39, 50, 38],
            'workclass': [7, 6, 4],
            'fnlwgt': [77516, 83311, 215646],
            'education': [9, 9, 11],
            'education-num': [13, 13, 9],
            'hours-per-week': [40, 13, 40],
            'target': [0, 0, 1]
        })
        st.dataframe(sample_df, use_container_width=True)
        st.caption("⚠️ **Note:** Categorical features should be encoded as integers")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p style='font-size: 1.1rem; font-weight: bold; margin-bottom: 0.5rem;'>
        💰 Adult Income Classification System
    </p>
    <p style='margin: 0.3rem 0;'>
        <strong>ML Assignment 2</strong> | TUSHAR KANTI SANTRA (2025AB05283)
    </p>
    <p style='margin: 0.3rem 0;'>
        BITS Pilani | Machine Learning Course
    </p>
    <p style='margin: 0.3rem 0;'>
        Dataset: <a href='https://archive.ics.uci.edu/dataset/2/adult' target='_blank'>UCI Machine Learning Repository</a>
    </p>
    <p style='margin-top: 1rem; font-size: 0.9rem;'>
        🔗 <a href='https://github.com/Tks008/ML_Classification_Streamlit_Assignment' target='_blank'>GitHub Repository</a>
    </p>
</div>
""", unsafe_allow_html=True)
