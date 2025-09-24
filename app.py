# app_day3_decision_tree.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Day 3 - Decision Tree (Titanic)", layout="wide")

st.title("🌳 titan survival predication - decision tree")
st.write(
    "Train & compare Decision Trees (Gini vs Entropy). Tune hyperparameters, "
    "visualize trees, feature importances, and try custom predictions."
)

# -------------------------
# Helper functions
# -------------------------
def preprocess(df):
    """Basic preprocessing: drop irrelevant columns, fill missing, encode."""
    df = df.copy()

    # Drop irrelevant columns if present
    df = df.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore")

    # Fill missing Age and Embarked; drop rows with missing Survived if any
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encode sex
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Embarked -> dummies (drop_first to avoid multicollinearity)
    if "Embarked" in df.columns:
        df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    # Ensure consistent feature columns
    # Choose feature list we will use:
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]

    # If some Embarked dummies are missing (e.g., dataset doesn't have that category),
    # create them with zeros
    for col in ["Embarked_Q", "Embarked_S"]:
        if col not in df.columns:
            df[col] = 0

    # Drop PassengerId if present
    if "PassengerId" in df.columns:
        df = df.drop(columns=["PassengerId"])

    return df, features

def show_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    # Rename class index
    df_report = df_report.rename(index={"0": "Not Survived", "1": "Survived"})
    # Display styled
    st.dataframe(df_report.style.background_gradient(cmap="Blues").format("{:.2f}"))

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Survived", "Survived"],
                yticklabels=["Not Survived", "Survived"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig)

# -------------------------
# Upload dataset
# -------------------------
uploaded = st.file_uploader("📂 Upload Titanic train.csv (Kaggle)", type=["csv"])
if uploaded is None:
    st.info("Upload the Titanic CSV (train.csv) to start. You can find it on Kaggle.")
    st.stop()

df_raw = pd.read_csv(uploaded)
st.success("Dataset uploaded ✅")
st.write("### Sample data")
st.dataframe(df_raw.head())

# -------------------------
# EDA
# -------------------------
st.header("🔍 Exploratory Data Analysis (EDA)")

col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Survival Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Survived", data=df_raw, palette="Set2", ax=ax)
    ax.set_xticklabels(["Not Survived", "Survived"])
    st.pyplot(fig)

with col2:
    st.subheader("Survival by Sex")
    fig, ax = plt.subplots()
    sns.countplot(x="Sex", hue="Survived", data=df_raw, palette="Set1", ax=ax)
    ax.set_ylabel("Count")
    ax.legend(["Not Survived", "Survived"])
    st.pyplot(fig)

col3, col4 = st.columns([1,1])
with col3:
    st.subheader("Survival by Pclass")
    fig, ax = plt.subplots()
    sns.countplot(x="Pclass", hue="Survived", data=df_raw, palette="pastel", ax=ax)
    ax.legend(["Not Survived", "Survived"])
    st.pyplot(fig)

with col4:
    st.subheader("Age Distribution by Survival")
    fig, ax = plt.subplots()
    # handle missing ages gracefully by dropping them for this plot
    sns.kdeplot(data=df_raw.dropna(subset=["Age"]), x="Age", hue="Survived", fill=True, ax=ax)
    st.pyplot(fig)

st.subheader("Correlation heatmap (numeric features)")
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(df_raw.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -------------------------
# Preprocess and prepare features
# -------------------------
st.header("⚙️ Preprocessing")

df, features = preprocess(df_raw)

st.write("Using features:", features)
if "Survived" not in df.columns:
    st.error("Uploaded CSV does not contain 'Survived' column. Make sure you uploaded train.csv from Kaggle.")
    st.stop()

X = df[features]
y = df["Survived"]

st.write("Data after preprocessing (first 5 rows):")
st.dataframe(pd.concat([X, y], axis=1).head())

# Train-test split
test_size = st.slider("Test set size (%)", 10, 40, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42, stratify=y)

st.write(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -------------------------
# Hyperparameters controls
# -------------------------
st.header("🔧 Model & Hyperparameter Tuning")

colg, cole = st.columns(2)
with colg:
    st.subheader("Gini Tree settings")
    gini_max_depth = st.slider("Gini: max_depth", 1, 20, 5, key="gini_depth")
    gini_min_samples_split = st.slider("Gini: min_samples_split", 2, 50, 2, key="gini_split")
with cole:
    st.subheader("Entropy Tree settings")
    ent_max_depth = st.slider("Entropy: max_depth", 1, 20, 5, key="ent_depth")
    ent_min_samples_split = st.slider("Entropy: min_samples_split", 2, 50, 2, key="ent_split")

# Button to train models
if st.button("▶️ Train & Compare Trees"):
    # Train Gini tree
    tree_gini = DecisionTreeClassifier(criterion="gini",
                                       max_depth=gini_max_depth,
                                       min_samples_split=gini_min_samples_split,
                                       random_state=42)
    tree_gini.fit(X_train, y_train)
    y_pred_gini = tree_gini.predict(X_test)
    acc_gini = accuracy_score(y_test, y_pred_gini)

    # Train Entropy tree
    tree_ent = DecisionTreeClassifier(criterion="entropy",
                                      max_depth=ent_max_depth,
                                      min_samples_split=ent_min_samples_split,
                                      random_state=42)
    tree_ent.fit(X_train, y_train)
    y_pred_ent = tree_ent.predict(X_test)
    acc_ent = accuracy_score(y_test, y_pred_ent)

    st.success("✅ Models trained")

    # Display accuracies
    c1, c2 = st.columns(2)
    c1.metric("Gini Tree Accuracy", f"{acc_gini:.2%}")
    c2.metric("Entropy Tree Accuracy", f"{acc_ent:.2%}")

    # Show confusion matrices side-by-side
    st.subheader("Confusion Matrices")
    cm1, cm2 = st.columns(2)
    with cm1:
        st.write("Gini")
        plot_confusion_matrix(confusion_matrix(y_test, y_pred_gini), title="Gini Confusion Matrix")
    with cm2:
        st.write("Entropy")
        plot_confusion_matrix(confusion_matrix(y_test, y_pred_ent), title="Entropy Confusion Matrix")

    # Classification reports
    st.subheader("Classification Reports")
    cr1, cr2 = st.columns(2)
    with cr1:
        st.write("Gini Report")
        show_classification_report(y_test, y_pred_gini)
    with cr2:
        st.write("Entropy Report")
        show_classification_report(y_test, y_pred_ent)

    # Feature importance
    st.subheader("Feature Importances (Gini)")
    fi = pd.Series(tree_gini.feature_importances_, index=features).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,3))
    sns.barplot(x=fi.values, y=fi.index, ax=ax)
    ax.set_xlabel("Importance")
    st.pyplot(fig)

    # Visualize trees (may be large)
    st.subheader("Decision Tree Visualization (Gini)")
    fig, ax = plt.subplots(figsize=(16,8))
    plot_tree(tree_gini, feature_names=features, class_names=["Not Survived","Survived"], filled=True, max_depth=3, fontsize=8, ax=ax)
    st.pyplot(fig)
    st.write("↳ (Displayed up to max_depth=3 for readability)")

    st.subheader("Decision Tree Visualization (Entropy)")
    fig, ax = plt.subplots(figsize=(16,8))
    plot_tree(tree_ent, feature_names=features, class_names=["Not Survived","Survived"], filled=True, max_depth=3, fontsize=8, ax=ax)
    st.pyplot(fig)

    # Tuning curve: effect of max_depth on train vs test accuracy (for one selected range)
    st.subheader("Overfitting / Underfitting: max_depth vs accuracy (Gini)")
    max_d_range = list(range(1, 16))
    train_acc = []
    test_acc = []
    for d in max_d_range:
        t = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        t.fit(X_train, y_train)
        train_acc.append(accuracy_score(y_train, t.predict(X_train)))
        test_acc.append(accuracy_score(y_test, t.predict(X_test)))

    fig, ax = plt.subplots()
    ax.plot(max_d_range, train_acc, label="Train Accuracy")
    ax.plot(max_d_range, test_acc, label="Test Accuracy")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Gini: Train vs Test Accuracy")
    ax.legend()
    st.pyplot(fig)

    # Option to save models
    if st.checkbox("Save trained models to disk (joblib)"):
        import joblib
        joblib.dump(tree_gini, "decision_tree_gini.pkl")
        joblib.dump(tree_ent, "decision_tree_entropy.pkl")
        st.success("Saved: decision_tree_gini.pkl and decision_tree_entropy.pkl")

    # Keep models in session state for prediction
    st.session_state["tree_gini"] = tree_gini
    st.session_state["tree_ent"] = tree_ent
    st.session_state["features"] = features

else:
    st.info("Adjust hyperparameters and click 'Train & Compare Trees' to run models.")
    # If models already in session_state, let user still use them for predictions
    if "tree_gini" in st.session_state:
        st.success("Loaded previously trained models from this session.")
        tree_gini = st.session_state["tree_gini"]
        tree_ent = st.session_state["tree_ent"]
        features = st.session_state["features"]
    else:
        tree_gini = tree_ent = None

# -------------------------
# Prediction form
# -------------------------
st.header("🔮 Try Prediction with Custom Input")

# Build input UI
col_a, col_b, col_c = st.columns(3)
with col_a:
    pclass = st.selectbox("Passenger Class (1=1st,2=2nd,3=3rd)", [1,2,3])
    sex = st.selectbox("Sex", ["male","female"])
    age = st.slider("Age", 0, 80, 30)
with col_b:
    sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 600.0, 32.2)
with col_c:
    embarked = st.selectbox("Embarked", ["C","Q","S"])
    model_choice = st.radio("Model for prediction", ("Gini", "Entropy"))

if st.button("Predict with Decision Tree"):
    # Ensure model exists
    model_to_use = None
    if model_choice == "Gini":
        model_to_use = st.session_state.get("tree_gini", None)
    else:
        model_to_use = st.session_state.get("tree_ent", None)

    if model_to_use is None:
        st.error("No trained model available. Train models first using 'Train & Compare Trees'.")
    else:
        sex_val = 0 if sex == "male" else 1
        embarked_q = 1 if embarked == "Q" else 0
        embarked_s = 1 if embarked == "S" else 0

        input_df = pd.DataFrame([[
            pclass, sex_val, age, sibsp, parch, fare, embarked_q, embarked_s
        ]], columns=features)

        pred = model_to_use.predict(input_df)[0]
        proba = model_to_use.predict_proba(input_df)[0] if hasattr(model_to_use, "predict_proba") else None

        if pred == 1:
            st.success(f"🎉 Predicted: SURVIVED (class=1){' — Probability: {:.2f}'.format(proba[1]) if proba is not None else ''}")
        else:
            st.error(f"☠️ Predicted: NOT SURVIVED (class=0){' — Probability: {:.2f}'.format(proba[0]) if proba is not None else ''}")

# -------------------------
# Documentation / Notes
# -------------------------
st.header("📘 Notes (Gini vs Entropy & Decision Tree Concepts)")
st.markdown(
"""
**Decision Tree** is a supervised learning algorithm that splits feature space into regions.
- **Gini Index** measures impurity: lower is better. Default in sklearn.
- **Entropy (Information Gain)** also measures impurity (based on information theory).
- Both often give similar results; entropy may be slightly slower due to log calculations.

**Overfitting vs Underfitting**
- Deep trees (high `max_depth`) often overfit: very high train accuracy but poor test accuracy.
- Shallow trees underfit: both train & test accuracies low.
- Use constraints (`max_depth`, `min_samples_split`, pruning) to regularize.

**Why Decision Tree for Titanic?**
- Handles categorical & numeric features without scaling.
- Provides interpretable decisions and feature importances.
"""
)

st.write("---")
st.write("End of app. Happy experimenting! 🚀")
