# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns

# st.set_page_config(page_title="Customer Segmentation", layout="wide")

# @st.cache_data
# def load_data():
#     # Upload from repo, or choose file in app
#     # For repo, adjust path if needed
#     return pd.read_csv("Mall_Customers.csv")

# df = load_data()

# st.title("🛍Customer Segmentation with KMeans Clustering")

# # Sidebar EDA options
# st.sidebar.header("Options")

# # ----- EDA -----
# st.header("1. Exploratory Data Analysis (EDA)")

# if st.checkbox("Show Raw Data"):
#     st.write(df)

# st.subheader("Basic Info")
# st.write(df.describe())

# col1, col2, col3 = st.columns(3)
# with col1:
#     st.write("### Gender Distribution")
#     gender_counts = df['Gender'].value_counts()
#     st.bar_chart(gender_counts)

# with col2:
#     st.write("### Age Distribution")
#     st.bar_chart(df['Age'])

# with col3:
#     st.write("### Annual Income Distribution")
#     st.bar_chart(df['Annual Income (k$)'])
    
# st.write("### Spending Score Distribution")
# fig, ax = plt.subplots()
# sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, ax=ax)
# st.pyplot(fig)

# # ----- Clustering -----
# st.header("2. KMeans Clustering")

# # Feature selection (keep simple: Income & Spending Score)
# features = ['Annual Income (k$)', 'Spending Score (1-100)']

# n_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 5)

# # Prepare data
# X = df[features]

# # KMeans
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# cluster_labels = kmeans.fit_predict(X)
# df['Cluster'] = cluster_labels

# # Visualize Clusters
# st.subheader("Clusters Visualization")
# fig2, ax2 = plt.subplots()
# scatter = ax2.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
#                       c=df['Cluster'], cmap='tab10', s=50)
# ax2.set_xlabel('Annual Income (k$)')
# ax2.set_ylabel('Spending Score (1-100)')
# ax2.set_title('KMeans Clusters')
# plt.colorbar(scatter, ax=ax2)
# st.pyplot(fig2)

# # Show cluster centers
# st.write("### Cluster Centers")
# centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
# centers["Cluster"] = range(n_clusters)
# st.dataframe(centers)

# # Clustered Data Preview
# st.write("### Sample of Clustered Data")
# st.dataframe(df.head())

# # Download clustered data
# st.sidebar.write("## Download Clustered Data")
# csv = df.to_csv(index=False).encode()
# st.sidebar.download_button("Download CSV", csv, "segmented_customers.csv", "text/csv")

# st.markdown("---")
# st.caption("Powered by Streamlit · KMeans Customer Segmentation · Demo App")


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# ---- THEME & BRANDING ----
# To customize theme, edit .streamlit/config.toml or use Streamlit Cloud UI
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
)

# ---- SIDEBAR NAVIGATION ----
st.sidebar.image("company_logo.png", use_column_width=True)  # Add your logo file
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("EDA", "Clustering", "Recommendation", "Download")
)

# ---- DATA HANDLING ----
@st.cache_data
def load_default_data():
    return pd.read_csv("Mall_Customers.csv")

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Your Customer CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!", icon="✔️")
else:
    df = load_default_data()

# ---- DYNAMIC FEATURE SELECTION ----
all_features = [col for col in df.columns if df[col].dtype in [np.number, 'float64', 'int64']] + [
    col for col in df.columns if df[col].nunique() < 10 and col != 'CustomerID']
default_features = ['Annual Income (k$)', 'Spending Score (1-100)']
st.session_state['features'] = st.sidebar.multiselect(
    "Select Features for Clustering",
    all_features,
    default=default_features,
)

# ---- TABS/DASHBOARD LAYOUT ----
st.title("🛍️ Customer Segmentation Dashboard")
st.markdown("A modern, interactive platform for actionable customer insights.")

if page == "EDA":
    tab1, tab2 = st.tabs(["Distributions", "Pairplot"])
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Gender Distribution")
            st.bar_chart(df['Gender'].value_counts())
        with c2:
            st.subheader("Age Distribution")
            st.bar_chart(df['Age'])
        with c3:
            st.subheader("Income Distribution")
            st.bar_chart(df['Annual Income (k$)'])
    with tab2:
        st.subheader("Bivariate Relationships")
        st.dataframe(df.head())
        st.caption("Pairplot and further EDA can be generated with additional code.")

elif page == "Clustering":
    st.header("Clustering Configuration")
    k = st.slider("Select Number of Clusters", 2, 10, 5)
    features = st.session_state['features'] or default_features

    # Encode 'Gender' for clustering if it's selected
    data_for_clustering = df[features].copy()
    if 'Gender' in data_for_clustering.columns:
        data_for_clustering['Gender'] = data_for_clustering['Gender'].map({'Male': 0, 'Female': 1})

    # Perform KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(data_for_clustering)

    # ---- CLUSTER LABELING ----
    if 'cluster_labels' not in st.session_state or st.session_state['cluster_labels'].get(k) is None:
        st.session_state['cluster_labels'] = {i: f"Cluster {i}" for i in range(k)}
        st.session_state['descriptions'] = {i: "" for i in range(k)}
    st.subheader("Rename Clusters and Add Descriptions")
    for i in range(k):
        col1, col2 = st.columns([1, 3])
        st.session_state['cluster_labels'][i] = col1.text_input(
            f"Label for Cluster {i}", st.session_state['cluster_labels'][i], key=f"label_{i}")
        st.session_state['descriptions'][i] = col2.text_area(
            f"Description for {st.session_state['cluster_labels'][i]}", st.session_state['descriptions'][i], key=f"desc_{i}")

    # Map display names to clusters
    df['Cluster Label'] = df['Cluster'].map(st.session_state['cluster_labels'])
    df['Cluster Description'] = df['Cluster'].map(st.session_state['descriptions'])

    st.success(f"Clustering completed with {k} clusters.")
    cluster_summary = df.groupby('Cluster Label').agg('mean')[features]
    st.write("### Cluster Centers (means):")
    st.dataframe(cluster_summary)

    st.write("### Clustering Visualization")
    if len(features) >= 2:
        import plotly.express as px
        fig = px.scatter(
            df,
            x=features[0],
            y=features[1],
            color='Cluster Label',
            hover_data=['Cluster Description'],
            symbol='Gender' if 'Gender' in features else None,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Select at least two features for meaningful visualization.")

elif page == "Recommendation":
    st.header("Segment Recommendations")
    st.write("Below are recommendations for each cluster segment (add business logic as needed):")
    for i in range(df['Cluster'].nunique()):
        label = st.session_state['cluster_labels'].get(i, f"Cluster {i}")
        desc = st.session_state['descriptions'].get(i, "")
        st.subheader(label)
        st.write(desc if desc else "No description provided.")
        # Placeholder: add business rules for recommendations here

elif page == "Download":
    st.header("Download Clustered Results")
    st.dataframe(df.head(10))
    csv_out = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Clustered Data as CSV",
        data=csv_out,
        file_name="clustered_customers.csv",
        mime='text/csv',
    )

st.sidebar.markdown("---")
st.sidebar.caption("Designed by [Your Name] | Powered by Streamlit")

# ---- OPTIONAL: Custom CSS ----
# Add below for advanced styling if desired
# st.markdown(
#     """
#     <style>
#     /* Custom CSS goes here */
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


