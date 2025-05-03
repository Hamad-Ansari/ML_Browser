# Libraries 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
# App heading
st.title('''# Explore ML Models and Datasets
## A Streamlit Web App for Machine Learning Model Comparison
## and Dataset Exploration
## Upload your dataset and explore different ML models
**With Hammad Ansari**''')

# Sidebar for dataset selection
dataset_option = st.sidebar.radio("Select Dataset Source", 
                                 ("Preloaded Datasets", "Upload Your Own"))

if dataset_option == "Preloaded Datasets":
    dataset_name = st.sidebar.selectbox("Select Dataset", 
                                      ("Iris", "Breast Cancer", "Wine"))
    uploaded_file = None
else:
    dataset_name = None
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")

# Classification algorithm selection
classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ("KNN", "SVM", "Random Forest")
)

# Function to get dataset
def get_dataset(dataset_option, dataset_name, uploaded_file):
    X, y = pd.DataFrame(), pd.DataFrame()
    
    if dataset_option == "Preloaded Datasets":
        if dataset_name == "Iris":
            iris = datasets.load_iris()
            X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
            y = pd.DataFrame(data=iris.target, columns=["target"])
        elif dataset_name == "Breast Cancer":
            breast_cancer = datasets.load_breast_cancer()
            X = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
            y = pd.DataFrame(data=breast_cancer.target, columns=["target"])
        elif dataset_name == "Wine":
            wine = datasets.load_wine()
            X = pd.DataFrame(data=wine.data, columns=wine.feature_names)
            y = pd.DataFrame(data=wine.target, columns=["target"])
    elif uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            if not data.empty:
                st.sidebar.subheader("Data Configuration")
                
                # Let user select target column
                target_col = st.sidebar.selectbox("Select target column", data.columns)
                
                # Separate features and target
                X = data.drop(columns=[target_col])
                y = data[[target_col]]
                
                # If target is not numeric, encode it
                if y[target_col].dtype == 'object':
                    le = LabelEncoder()
                    y[target_col] = le.fit_transform(y[target_col])
                    st.sidebar.write("Note: Target column has been label encoded")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    return X, y

# Get dataset based on user selection
if dataset_option == "Preloaded Datasets" or (dataset_option == "Upload Your Own" and uploaded_file is not None):
    X, y = get_dataset(dataset_option, dataset_name, uploaded_file)
else:
    X, y = pd.DataFrame(), pd.DataFrame()

if not X.empty:
    # Dataset info
    st.write("### Dataset Information")
    st.write("**Shape of dataset:**", X.shape)
    st.write("**Number of classes:**", len(np.unique(y)))
    
    # Show sample data
    if st.checkbox("Show raw data"):
        st.write("### Sample Data")
        st.write(pd.concat([X, y], axis=1).head())

    # Define the different classifier parameters
    def add_parameter_ui(classifier_name):
        params = {}
        if classifier_name == "KNN":
            K = st.sidebar.slider("K", 1, 15)
            params["K"] = K
        elif classifier_name == "SVM":
            C = st.sidebar.slider("C", 0.01, 10.0)
            params["C"] = C
        else:
            max_depth = st.sidebar.slider("max_depth", 2, 15)
            n_estimators = st.sidebar.slider("n_estimators", 1, 100)
            params["max_depth"] = max_depth
            params["n_estimators"] = n_estimators
        return params

    # Get classifier parameters
    params = add_parameter_ui(classifier_name)

    # Function for classifier
    def get_classifier(classifier_name, params):
        if classifier_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=params["K"])
        elif classifier_name == "SVM":
            clf = SVC(C=params["C"])
        else:
            clf = RandomForestClassifier(
                n_estimators=params["n_estimators"], 
                max_depth=params["max_depth"], 
                random_state=1234
            )
        return clf

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # Fit the model
    clf = get_classifier(classifier_name, params)
    #clf.fit(X_train, y_train.values.ravel())
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    y_pred = clf.fit(X_train, y_train.values.ravel()).predict(X_test)
    #y_pred = clf.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=["target"])
    y_test = pd.DataFrame(y_test.values.ravel(), columns=["target"])
    #y_test = pd.DataFrame(y_test.values.ravel(), columns=["target"])
    
    le = LabelEncoder()  # Initialize LabelEncoder
   # X_train['Gender'] = le.fit_transform(X_train['Gender'])  # 'Male' -> 1, 'Female' -> 0
    y_pred = clf.predict(X_test)
    print(X_train.dtypes)
    print(X_train.head())

    
    # Model evaluation
    st.write("### Model Evaluation")
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Classifier:** {classifier_name}")
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write("**Parameters:**", params)
    
    # Confusion matrix
    st.write("#### Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)
    
    # Classification report
    st.write("#### Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Data Visualization
    st.write("### Data Visualization")
    
    # Plot original data (first two features)
    if X.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y.iloc[:, 0], cmap="viridis", s=50)
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        ax.set_title("First Two Features")
        st.pyplot(fig)
    
    # Plot the data set with PCA
    try:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y.iloc[:, 0], cmap="viridis", s=50)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("PCA Visualization")
        st.pyplot(fig)
        
        # Plot with classifier predictions
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y.iloc[:, 0], cmap="viridis", s=50)
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clf.predict(X), cmap="coolwarm", s=50, alpha=0.3)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("Classifier Predictions")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not create PCA visualization: {str(e)}")

else:
    if dataset_option == "Upload Your Own":
        st.info("Please upload a CSV file to get started")
    else:
        st.info("Please select a dataset from the sidebar")