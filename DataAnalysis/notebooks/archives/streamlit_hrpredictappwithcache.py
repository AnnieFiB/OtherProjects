# streamlit_hrpredictapp.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from kaggle.api.kaggle_api_extended import KaggleApi
import requests
from io import StringIO
from zipfile import ZipFile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from hrattrition import *

# Constants
TARGET = 'Attrition'
targets = 'attrition'
MODEL_PATH = "attrition_best_model.pkl"
PIPELINE_PATH = "preprocessing_pipeline.pkl"
DATASET_REF = "raneemoqaily/ibm-hr-analytics-employee-attrition-performance"
FILE_NAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
GITHUB_URL = "https://raw.githubusercontent.com/IBM-Data-Science-Professional-Certificate/ibm-hr-analytics-attrition/master/WA_Fn-UseC_-HR-Employee-Attrition.csv"

@st.cache_data(show_spinner=False)
def cached_preprocessing(_cleaned_data, target):
    """Cache feature engineering and splitting"""
    X, y = prepare_features(_cleaned_data, target)
    return scale_and_split(_cleaned_data, target)

@st.cache_data(show_spinner=False)
def cached_feature_importance(_cleaned_data, target):
    """Cache feature importance calculations"""
    return compute_combined_feature_importance(_cleaned_data, target_column=target, plot=False)

@st.cache_resource(show_spinner=False)
def cached_model_training(X_train, y_train):
    """Cache model training results"""
    return train_and_evaluate_models(X_train, y_train)

@st.cache_resource(show_spinner=False) 
def cached_model_tuning(X_train, y_train, models_and_params):
    """Cache hyperparameter tuning results"""
    return tune_and_select_best_model(X_train, y_train, models_and_params)

# -----------------------------
# Data Loading
# -----------------------------
def setup_kaggle():
    """Configure Kaggle API with clear user guidance"""
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        st.sidebar.markdown("### 🔑 Kaggle API Setup")
        st.sidebar.markdown("""
        1. Go to [Kaggle Settings](https://www.kaggle.com/settings)
        2. Scroll to 'API' section
        3. Click 'Create New API Token'
        4. Upload the downloaded `kaggle.json` below
        """)
        uploaded = st.sidebar.file_uploader("Upload kaggle.json", type="json", label_visibility="collapsed")
        if uploaded:
            os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
            with open(os.path.expanduser("~/.kaggle/kaggle.json"), "wb") as f:
                f.write(uploaded.getvalue())
            os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
            st.sidebar.success("API configured! Please restart the app")
            st.stop()
        return False
    return True

@st.cache_data(ttl=3600)
def load_data():
    """Robust data loader with multiple fallbacks"""
    try:
        if setup_kaggle():
            api = KaggleApi()
            api.authenticate()
            with st.spinner("⏳ Downloading dataset from Kaggle..."):
                api.dataset_download_files(DATASET_REF, unzip=True)
                if os.path.exists(FILE_NAME):
                    df = pd.read_csv(FILE_NAME)
                    st.session_state.raw_data = df.copy()
                    st.success("✅ Dataset downloaded successfully!")
                    return df
                raise Exception("File not found after download")
    except Exception as e:
        st.warning(f"Kaggle download failed: {str(e)}")
    
    # GitHub fallback
    with st.spinner("🔄 Attempting GitHub download..."):
        try:
            response = requests.get(GITHUB_URL)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            st.session_state.raw_data = df.copy()
            st.success("✅ Dataset loaded from GitHub")
            return df
        except:
            st.error("⚠️ GitHub download failed")
    
    # Local fallback
    if os.path.exists("local_attrition_data.csv"):
        df = pd.read_csv("local_attrition_data.csv")
        st.session_state.raw_data = df.copy()
        return df
    
    # Sample data
    st.warning("Using sample data - limited functionality")
    df = pd.DataFrame({
        'Age': [32, 28, 45], 
        'Attrition': ['No', 'Yes', 'No'],
        'MonthlyIncome': [5000, 3000, 7000]
    })
    st.session_state.raw_data = df.copy()
    return df

# -----------------------------
# Main Application
# -----------------------------
def main():
    st.set_page_config("IBM HR Analytics", layout="wide", page_icon="💼")
    st.title("Employee Attrition Analysis Dashboard")

    # Load and clean data
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = load_data()
    if 'cleaned_data' not in st.session_state and not st.session_state.raw_data.empty:
        with st.spinner("🧼 Cleaning data..."):
            config = detect_columns(st.session_state.raw_data)
            st.session_state.cleaned_data = clean_data(st.session_state.raw_data.copy(), config)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 Data Overview", "📊 Statistical Analysis", "🔗 Correlation Insights",
        "🤖 Predictive Modeling", "📈 Retention Strategies", "🔮 Predict New Data"
    ])

    with tab1:
        st.header("Data Overview")
        if not st.session_state.raw_data.empty:
            st.subheader("Raw Data Preview")
            st.dataframe(st.session_state.raw_data.head())
            
            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.cleaned_data.head())
            
            st.subheader("Data Summary")
            st.write(f"Total Employees: {len(st.session_state.cleaned_data)}")
            attrition_rate = st.session_state.cleaned_data[targets].mean()
            st.metric("Attrition Rate", f"{attrition_rate:.1%}")

    with tab2:
        st.header("Statistical Analysis")
        if 'cleaned_data' in st.session_state:
            cleaned_data = st.session_state.cleaned_data
            
            with st.expander("Categorical Features (Chi-Square)"):
                cat_cols = cleaned_data.select_dtypes(include=['object', 'category']).columns.tolist()
                if targets in cat_cols:
                    cat_cols.remove(targets)
                if cat_cols:
                    chi_results = chi_square_test_batch(cleaned_data, cat_cols, targets)
                    st.dataframe(chi_results.sort_values('p_value'))
                    
                    sig_features = chi_results[chi_results['significant']]['feature'].tolist()
                    if sig_features:
                        st.subheader("Significant Categorical Features")
                        figures = plot_significant_categorical_proportions(cleaned_data, sig_features, targets)
                        for fig in figures:
                            st.pyplot(fig)
                            plt.close(fig)

            with st.expander("Numerical Features (ANOVA)"):
                num_cols = cleaned_data.select_dtypes(include=np.number).columns.tolist()
                if targets in num_cols:
                    num_cols.remove(targets)
                
                if num_cols:
                    valid_features = []
                    constant_features = []
                    
                    for feat in num_cols:
                        group0 = cleaned_data[cleaned_data[targets] == 0][feat]
                        group1 = cleaned_data[cleaned_data[targets] == 1][feat]
                        
                        if group0.nunique() > 1 and group1.nunique() > 1:
                            valid_features.append(feat)
                        else:
                            constant_features.append(feat)
                    
                    if constant_features:
                        st.warning(f"""
                        **Constant features skipped:**  
                        {', '.join(constant_features)}  
                        These features have no variation within attrition groups
                        """)
                    
                    if valid_features:
                        anova_results = anova_test_numerical_features(cleaned_data, valid_features, targets)
                        st.dataframe(anova_results.sort_values('p_value'))
                        
                        sig_num_features = anova_results[anova_results['significant']]['feature'].tolist()
                        if sig_num_features:
                            st.subheader("Top Significant Numerical Features")
                            for feat in sig_num_features[:20]:
                                fig = plt.figure(figsize=(10, 4))
                                sns.boxplot(data=cleaned_data, x=targets, y=feat)
                                plt.title(f"{feat} by {targets}")
                                st.pyplot(fig)
                                plt.close(fig)
                    else:
                        st.error("No valid numerical features for ANOVA - all features are constant within groups")

            with st.expander("**Key Findings**"):
                st.markdown("""
                **🧍 Demographic Factors**  
                - **Age:** Younger employees are significantly more likely to leave the organization  
                - **Marital Status:** Single employees show the highest attrition rate compared to married or divorced employees  
                - **Distance from Home:** Employees living farther from work are more likely to leave  

                **💼 Professional Factors**  
                - **Job Role:** Sales Representatives and Lab Technicians have notably higher attrition  
                - **Department:** Sales shows the highest attrition; R&D the lowest  
                - **Job Level:** Lower job levels are associated with higher attrition  
                - **Business Travel:** Frequent travelers are more likely to leave  
                - **OverTime:** Employees working overtime show a strong tendency to attrite  
                - **Total Working Years:** Employees with less experience are more prone to leave  

                **🧠 Behavioral Factors**  
                - **Job Satisfaction:** Lower satisfaction scores are significantly associated with higher attrition  
                - **Environment Satisfaction:** Poor work environment correlates with turnover  
                - **Job Involvement:** Reduced engagement links to increased attrition  
                - **Training Times Last Year:** Employees receiving less training show higher turnover  
                - **Work-Life Balance:** Lower balance correlates with increased attrition  
                - **Years at Company:** Shorter tenures strongly link to higher attrition  
                - **Years With Manager:** Less time with current manager correlates with turnover  
                """)
                
    with tab3:
        st.header("Correlation Analysis")
        if 'cleaned_data' in st.session_state:
            cleaned_data = st.session_state.cleaned_data
            num_cols = cleaned_data.select_dtypes(include=np.number).columns.tolist()
            
            fig = plt.figure(figsize=(12, 8))
            sns.heatmap(cleaned_data[num_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
            st.pyplot(fig)
            plt.close(fig)

        with st.expander("**Key Findings**"):
            st.markdown("""
                - Job Satisfaction, Environment Satisfaction, and Job Involvement are negatively correlated with attrition. Employees with lower scores in these areas are more likely to leave.
                - Monthly Income shows a moderate negative correlation with attrition. Lower income brackets are at higher risk.
                - OverTime is positively correlated with attrition. Excessive work hours may lead to burnout.
                - Tenure-related features (YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager) are negatively correlated with attrition. Employees often leave within their early years.
                
                These correlations support prioritizing compensation fairness, job engagement, and early employee experience in attrition mitigation strategies.
                
                """)
    with tab4:
        st.header("Predictive Modeling")
        if 'cleaned_data' in st.session_state:
            cleaned_data = st.session_state.cleaned_data
            
            if st.button("🎯 Start Modeling Pipeline"):
                with st.status("🔨 Data Preparation", expanded=True) as status:
                    with st.spinner("Preprocessing features..."):
                        #X, y = prepare_features(cleaned_data, targets)
                        X_train, X_test, y_train, y_test, pipeline = cached_preprocessing(cleaned_data, targets)
                        st.session_state.pipeline = pipeline
                        st.success("Features preprocessed")
                    
                    with st.spinner("Analyzing class distribution..."):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original Distribution**")
                            orig_dist = pd.DataFrame({
                                'Class': ['Retained', 'Attrited'],
                                'Count': [sum(y == 0), sum(y == 1)]
                            })
                            st.dataframe(orig_dist)
                            fig = px.pie(orig_dist, names='Class', values='Count', 
                                       title="Original Class Distribution")
                            st.plotly_chart(fig)
                        
                    with st.spinner("Balancing classes with SMOTE..."):
                        X_train_res, y_train_res = balance_classes_smote(X_train, y_train)
                        with col2:
                            st.markdown("**Balanced Distribution**")
                            balanced_dist = pd.DataFrame({
                                'Class': ['Retained', 'Attrited'],
                                'Count': [sum(y_train_res == 0), sum(y_train_res == 1)]
                            })
                            st.dataframe(balanced_dist)
                            fig = px.pie(balanced_dist, names='Class', values='Count',
                                       title="Balanced Class Distribution")
                            st.plotly_chart(fig)
                        status.update(label="Data preparation complete", state="complete")

                with st.status("📊 Feature Importance Analysis", expanded=True) as status:
                    try:
                        st.subheader("Combined Feature Importance")
                        importance_df = cached_feature_importance(
                            cleaned_data, 
                            target_column=targets,
                            top_n=15,
                            plot=False
                        )
                        
                        # Create Plotly figure
                        fig = px.bar(
                            importance_df.head(15),
                            x=['importance_rf', 'importance_mi'],
                            y='feature',
                            barmode='group',
                            title='Feature Importances: Random Forest vs Mutual Information',
                            labels={'value': 'Importance Score'},
                            color_discrete_sequence=['teal', 'coral']
                        )
                        fig.update_layout(
                            yaxis={'categoryorder':'total ascending'},
                            height=500,
                            legend_title='Metric'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        status.update(label="Feature analysis complete", state="complete")
                    except Exception as e:
                        st.error(f"Feature importance analysis failed: {str(e)}")
                        status.update(label="Feature analysis failed", state="error")

                with st.status("🤖 Model Training", expanded=True) as status:
                    with st.spinner("Training base models..."):
                        results, models = cached_model_training(X_train_res, y_train_res)
                        
                        st.subheader("Model Performance Comparison")
                        metrics_df = results_to_dataframe(results)
                        fig = px.bar(metrics_df, 
                                   x='Model', y='Recall (1)',
                                   title='Attrition Recall by Model')
                        st.plotly_chart(fig)
                        st.dataframe(metrics_df.style.format({
                            'ROC AUC': '{:.3f}',
                            'Accuracy': '{:.1%}',
                            'Precision (1)': '{:.1%}',
                            'Recall (1)': '{:.1%}'
                        }))
                        status.update(label="Base models trained", state="complete")

                with st.status("⚙️ Hyperparameter Tuning", expanded=True) as status:
                    with st.spinner("Optimizing models..."):
                        models_and_params = {
                            "Random Forest": (
                                RandomForestClassifier(class_weight="balanced"),
                                {'n_estimators': [100, 200], 
                                 'max_depth': [5, 10],
                                 'min_samples_split': [2, 5]}
                            ),
                            "SVM": (
                                SVC(probability=True, class_weight="balanced"),
                                {'C': [0.1, 1, 10], 
                                 'kernel': ['linear', 'rbf'],
                                 'gamma': ['scale', 'auto']}
                            )
                        }
                        
                        try:
                            best_model, best_name, best_scores = cached_model_tuning(
                                X_train_res, y_train_res, models_and_params
                            )
                        except Exception as e:
                            st.error(f"Tuning failed: {str(e)}")
                            st.stop()
                        
                        st.session_state.best_model = best_model
                        st.success(f"Optimization complete! Best model: {best_name}")
                        
                        st.subheader("Optimized Performance Metrics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Best Recall", f"{best_scores['recall']:.1%}")
                        with col2:
                            st.metric("Best F1 Score", f"{best_scores['f1']:.1%}")
                        with col3:
                            st.metric("Model Type", best_name)
                            
                        st.write("Best Parameters:")
                        st.json(best_scores['best_params'])
                        
                        status.update(label="Hyperparameter tuning complete", state="complete")

                st.balloons()
                st.success("🎉 Modeling pipeline completed successfully!")

    with tab5:
        st.header("Retention Strategies")
        st.markdown("""
        **Key Recommendations:**
        1. Workload Management: Monitor overtime patterns; proactively check in with high-overtime employees.
        2. Satisfaction Surveys: Implement early and frequent pulse surveys for satisfaction, especially within first 1-2 years.
        3. Career Development: Offer structured growth plans and internal mobility to avoid early stagnation.
        4. Compensation Review:	Benchmark and adjust salaries for lower-level employees at high risk.
        5. Onboarding & Tenure	Identify and support new joiners in their first 2 years — when attrition is highest.
        6. Commute Flexibility	Consider hybrid/work-from-home options for those with long commute distances.
        7. Establish clear promotion paths for junior staff
        

         **Key Metrics to Monitor Going Forward**
        - Attrition by Role/Level — Track churn within specific job roles and job levels.
        - OverTime Trends — Flag employees regularly logging extra hours.
        - Engagement Metrics — Track satisfaction and involvement scores over time.
        - Tenure Benchmarks — Monitor attrition within first 2–3 years of employment.
        - Compensation Distribution — Keep salary equity dashboards across similar roles/levels.
        - Promotion Gaps — Identify employees who haven't been promoted in 3+ years.

         **Success Metrics:**
        - Target attrition reduction: 35-45%
        - Expected cost savings: $1.2M - $1.8M annually
        """)

    with tab6:
        st.header("Predict New Data")
        if 'best_model' in st.session_state and 'pipeline' in st.session_state:
            uploaded_file = st.file_uploader("Upload CSV for predictions", type="csv")
            if uploaded_file:
                try:
                    new_data = pd.read_csv(uploaded_file)
                    processed = st.session_state.pipeline.transform(new_data)
                    predictions = st.session_state.best_model.predict(processed)
                    new_data['Attrition Risk'] = ['High' if p == 1 else 'Low' for p in predictions]
                    st.success("Predictions complete!")
                    st.dataframe(new_data.style.applymap(
                        lambda x: 'background-color: #ffcccc' if x == 'High' else '',
                        subset=['Attrition Risk']
                    ))
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
        else:
            st.warning("Please train models first in the Predictive Modeling tab")

if __name__ == "__main__":
    main()