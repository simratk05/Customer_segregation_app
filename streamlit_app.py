import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation", layout="wide")

@st.cache_data
def load_data():
    # Upload from repo, or choose file in app
    # For repo, adjust path if needed
    return pd.read_csv("Mall_Customers.csv")

df = load_data()

st.title("🛍️ Customer Segmentation with KMeans Clustering")

# Sidebar EDA options
st.sidebar.header("Options")

# ----- EDA -----
st.header("1. Exploratory Data Analysis (EDA)")

if st.checkbox("Show Raw Data"):
    st.write(df)

st.subheader("Basic Info")
st.write(df.describe())

col1, col2, col3 = st.columns(3)
with col1:
    st.write("### Gender Distribution")
    gender_counts = df['Gender'].value_counts()
    st.bar_chart(gender_counts)

with col2:
    st.write("### Age Distribution")
    st.bar_chart(df['Age'])

with col3:
    st.write("### Annual Income Distribution")
    st.bar_chart(df['Annual Income (k$)'])
    
st.write("### Spending Score Distribution")
fig, ax = plt.subplots()
sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# ----- Clustering -----
st.header("2. KMeans Clustering")

# Feature selection (keep simple: Income & Spending Score)
features = ['Annual Income (k$)', 'Spending Score (1-100)']

n_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 5)

# Prepare data
X = df[features]

# KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X)
df['Cluster'] = cluster_labels

# Visualize Clusters
st.subheader("Clusters Visualization")
fig2, ax2 = plt.subplots()
scatter = ax2.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                      c=df['Cluster'], cmap='tab10', s=50)
ax2.set_xlabel('Annual Income (k$)')
ax2.set_ylabel('Spending Score (1-100)')
ax2.set_title('KMeans Clusters')
plt.colorbar(scatter, ax=ax2)
st.pyplot(fig2)

# Show cluster centers
st.write("### Cluster Centers")
centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
centers["Cluster"] = range(n_clusters)
st.dataframe(centers)

# Clustered Data Preview
st.write("### Sample of Clustered Data")
st.dataframe(df.head())

# Download clustered data
st.sidebar.write("## Download Clustered Data")
csv = df.to_csv(index=False).encode()
st.sidebar.download_button("Download CSV", csv, "segmented_customers.csv", "text/csv")

st.markdown("---")
st.caption("Powered by Streamlit · KMeans Customer Segmentation · Demo App")

