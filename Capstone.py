import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(page_title="BEM&T CPK", page_icon="📊", layout="wide")
st.title("📊 BEM&T CPK")

# File uploader
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

# Function to load data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, dtype=str)
        return df
    return None

# Function to create a chart for data visualization with user selection in sidebar
def plot_custom_chart(df):
    st.sidebar.subheader("Customize Your Graph")
    chart_type = st.sidebar.radio("Select Graph Type:", ["Bar", "Pie"], index=0)
    columns = df.columns.tolist()
    x_axis = st.sidebar.selectbox("Choose X-axis:", columns, index=0)
    y_axis = st.sidebar.selectbox("Choose Y-axis:", columns, index=1)
    filter_column = st.sidebar.selectbox("Choose Filter Column (Optional):", [None] + columns)
    filter_value = None
    if filter_column:
        filter_values = df[filter_column].unique()
        filter_value = st.sidebar.selectbox("Choose Filter Value:", filter_values)
    
    if filter_value:
        df = df[df[filter_column] == filter_value]
    
    df = df.groupby(x_axis)[y_axis].count().reset_index()  # Ensure proper count
    df = df.sort_values(by=x_axis, ascending=True)  # Sort x-axis in ascending order
    
    if chart_type == "Bar":
        fig = px.bar(df, x=x_axis, y=y_axis, text=y_axis)
    elif chart_type == "Pie":
        df[y_axis] = df[y_axis].astype(int)  # Ensure numerical values for pie chart
        fig = px.pie(df, names=x_axis, values=y_axis, 
                     labels={x_axis: "Category", y_axis: "Count"}, hole=0.3)
        fig.update_traces(textinfo='label+percent', textposition='outside')
    
    st.plotly_chart(fig)

# Streamlit UI for Data Visualization
st.header("Data Visualization")
df = load_data(uploaded_file)
if df is not None:
    plot_custom_chart(df)

# Function to process data and extract key findings
def analyze_low_cpk(df):
    df['Final CPK GRP'] = df['Final CPK GRP'].astype(str).str.strip()
    df_low_cpk = df[df['Final CPK GRP'] == "<1.00"]
    
    if df_low_cpk.empty:
        return None
    
    analysis = df_low_cpk.groupby(['Plant', 'Process Step', 'Parameter']).size().reset_index(name='Count')
    return analysis

# Function to plot bar chart for low CPK data
def plot_bar_chart(analysis):
    fig, ax = plt.subplots(figsize=(10, 5))
    analysis_sorted = analysis.sort_values(by='Count', ascending=False)
    ax.barh(analysis_sorted['Process Step'] + ' - ' + analysis_sorted['Parameter'], analysis_sorted['Count'])
    ax.set_xlabel("Occurrences")
    ax.set_ylabel("Process Step - Parameter")
    ax.set_title("Most Frequent Issues (Final CPK GRP <1.00)")
    st.pyplot(fig)

# Function to train machine learning model
def train_ml_model(df):
    df['Final CPK GRP'] = df['Final CPK GRP'].astype(str).str.strip()
    df['Target'] = (df['Final CPK GRP'] == "<1.00").astype(int)  # Convert to binary classification
    
    features = ['Plant', 'Process Step', 'Parameter']
    df = df.dropna(subset=features + ['Target'])
    
    label_encoders = {}
    for col in features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return model, accuracy, label_encoders

# Streamlit UI for Final CPK GRP Analysis & Prediction
st.title("Final CPK GRP Analysis & Prediction")
if df is not None:
    analysis = analyze_low_cpk(df)
    if analysis is not None:
        st.subheader("Key Findings")
        st.dataframe(analysis)
        
        st.subheader("Bar Chart of Most Frequent Issues")
        plot_bar_chart(analysis)
    else:
        st.warning("No rows found with Final CPK GRP <1.00")
    
    st.subheader("Machine Learning Model Training")
    model, accuracy, label_encoders = train_ml_model(df)
    st.write(f"Model Accuracy: {accuracy:.2%}")
    
    st.subheader("Predict Future Issues")
    plant = st.text_input("Enter Plant:")
    process_step = st.text_input("Enter Process Step:")
    parameter = st.text_input("Enter Parameter:")
    
    if st.button("Predict"):  
        if plant and process_step and parameter:
            try:
                encoded_input = [
                    label_encoders['Plant'].transform([plant])[0],
                    label_encoders['Process Step'].transform([process_step])[0],
                    label_encoders['Parameter'].transform([parameter])[0]
                ]
                prediction = model.predict([encoded_input])[0]
                result = "Likely to have CPK <1.00" if prediction == 1 else "Unlikely to have CPK <1.00"
                st.write(f"Prediction: {result}")
            except KeyError:
                st.error("Entered values not found in the trained model. Please check your input values.")
        else:
            st.warning("Please enter all fields for prediction.")
else:
    st.info("Upload an Excel file to begin analysis.")
