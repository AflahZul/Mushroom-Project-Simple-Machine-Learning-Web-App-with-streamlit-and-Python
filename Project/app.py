import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# New correct imports for plots
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score
)

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title('Binary Classification Web App')
    st.markdown("### Are your mushrooms edible or poisonous?")
    st.sidebar.markdown("### Are your mushrooms edible or poisonous?")

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('/home/coder/Desktop/Project/mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache_data(persist=True)
    def split(df):
        y = df['type']                    # 'type' column: e=edible, p=poisonous
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0, stratify=y
        )
        return x_train, x_test, y_train, y_test

    # Fixed plotting function using the new scikit-learn API
    def plot_metrics(metrics_list, model, x_test, y_test, class_names):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_estimator(
                model, x_test, y_test,
                display_labels=class_names,
                cmap='Blues',
                ax=ax
            )
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            plt.close(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader('ROC Curve')
            fig, ax = plt.subplots(figsize=(6, 5))
            RocCurveDisplay.from_estimator(
                model, x_test, y_test,
                ax=ax
            )
            ax.set_title('ROC Curve')
            st.pyplot(fig)
            plt.close(fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            fig, ax = plt.subplots(figsize=(6, 5))
            PrecisionRecallDisplay.from_estimator(
                model, x_test, y_test,
                ax=ax
            )
            ax.set_title('Precision-Recall Curve')
            st.pyplot(fig)
            plt.close(fig)

    # Load data
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']   # Correct order: 0=edible, 1=poisonous

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier",
        ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest")
    )

    # ====================== SVM ======================
    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, step=0.01, value=1.0, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
        )

        if st.sidebar.button("Classify", key="classify_svm"):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)  # probability=True for ROC/PR
            model.fit(x_train, y_train)

            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            st.write("Accuracy:", accuracy.round(4))
            st.write("Precision:", precision_score(y_test, y_pred, pos_label=1).round(4))
            st.write("Recall:", recall_score(y_test, y_pred, pos_label=1).round(4))

            plot_metrics(metrics, model, x_test, y_test, class_names)

    # ====================== Logistic Regression ======================
    elif classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, step=0.01, value=1.0, key='C_LR')
        max_iter = st.sidebar.slider("Maximum iterations", 100, 500, 200, key='max_iter')

        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
            key='metrics_lr'
        )

        if st.sidebar.button("Classify", key="classify_lr"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)

            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            st.write("Accuracy:", accuracy.round(4))
            st.write("Precision:", precision_score(y_test, y_pred, pos_label=1).round(4))
            st.write("Recall:", recall_score(y_test, y_pred, pos_label=1).round(4))

            plot_metrics(metrics, model, x_test, y_test, class_names)

       # ====================== Random Forest ======================
    elif classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        
        n_estimators = st.sidebar.slider("Number of trees in the forest", 100, 5000, 100, key='n_estimators')
        max_depth = st.sidebar.slider("Maximum depth of each tree", 1, 30, 1, key='max_depth')  # None = unlimited
        min_samples_split = st.sidebar.slider("Min samples required to split a node", 2, 20, 2, key='min_samples_split')
        min_samples_leaf = st.sidebar.slider("Min samples required at a leaf node", 1, 20, 1, key='min_samples_leaf')
        
        # Bootstrap toggle — this is what you asked for!
        bootstrap = st.sidebar.radio("Bootstrap samples?", ("Yes (recommended)", "No"), index=0, key='bootstrap')
        bootstrap = True if bootstrap == "Yes (recommended)" else False

        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
            key='metrics_rf'
        )

        if st.sidebar.button("Classify", key="classify_rf"):
            st.subheader("Random Forest Results")
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,           # This controls bagging vs pasting
                random_state=0,
                n_jobs=-1                      # Use all CPU cores
            )
            
            model.fit(x_train, y_train)

            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            st.write("**Accuracy:**", f"{accuracy:.4f}")
            st.write("**Precision:**", f"{precision_score(y_test, y_pred, pos_label=1):.4f}")
            st.write("**Recall:**", f"{recall_score(y_test, y_pred, pos_label=1):.4f}")

            # Show whether bootstrap was used
            st.info(f"Bootstrap sampling: **{'Enabled' if bootstrap else 'Disabled (Pasting)'}**")

            plot_metrics(metrics, model, x_test, y_test, class_names)

if __name__ == "__main__":
    main()