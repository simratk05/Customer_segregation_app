# Customer Segmentation App

This app is a multi-page interactive Streamlit dashboard for customer segmentation based on clustering (KMeans). It supports data upload, real-time clustering config, cluster labeling, and download of results. The dataset for this app was taken from Kaggle

URL= https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python/data

The user can uplaod its own data, and get the clustered results. 

PRO FEATURE= Users can download the csv file of the results.

ACCESS THE APP FROM HERE- https://customersegregationapp0simrat.streamlit.app/

## Purpose

Gain actionable insights into your customer base using **KMeans Clustering**. Segment your market, identify target groups, and craft informed strategies with intuitive, no-code workflows.


## Feature Overview

- **Custom Sidebar Navigation:** Navigate between EDA, Clustering, Recommendations, and Download pages with a polished sidebar menu.
  
- **Modern Dashboard Layout:** Clean visuals using Streamlit’s columns, tabs, and branded theming.

- **Reusable Data:** Upload your own CSV or use the provided dataset for instant results.

- **Dynamic Feature Selection:** Cluster on any combination of customer attributes—Age, Gender, Income, Spending Score, etc.
  
- **Interactive, Labelled Clusters:** Name and describe segments with your own business logic and context.

- **One-Click Export:** Download the full, clustered dataset for further analysis.


## Workflow of the App-

1. **Landing Page**
   - App loads the default dataset or lets you upload your own customer CSV.
     
     <img width="1914" height="906" alt="image" src="https://github.com/user-attachments/assets/2730b4e1-fe6f-4832-90c0-7618dc3d6746" />


2. **Explore with EDA**
   - Go to the `EDA` tab to understand your data.
      - View distribution charts for gender, age, income, and spending.

3. **Run Clustering**
   - In the `Clustering` section:
      - Select which features to cluster on via the sidebar.
    <img width="1919" height="881" alt="image" src="https://github.com/user-attachments/assets/2af8e851-ae0b-4b10-a2c8-d4680130c541" />

      - Set the number of desired clusters.
      - Hit the controls; clusters are computed instantly.
      - Interactively rename each cluster and add business-context descriptions.
        <img width="1471" height="782" alt="image" src="https://github.com/user-attachments/assets/e9680839-96e3-403e-b374-72f0738e24cf" />

      - Review live, interactive scatter plots (Plotly-powered) with color-coded clusters.
        <img width="1473" height="787" alt="image" src="https://github.com/user-attachments/assets/953ccaaf-66dd-4483-b2c1-ac34bbea7158" />


4. **Export Results**
   - Go to `Download`:
      - Preview your data with cluster labels and descriptions.
      - One-click to download the enriched CSV for your CRM or further modeling.
<img width="1495" height="838" alt="image" src="https://github.com/user-attachments/assets/ff75182b-6fba-40dc-b08c-1996cf1c1d34" />


## 📥 File Upload

- Use your own `.csv` data—headers must include customer ID and relevant features.
- The app adapts to variable column names as long as they exist in the table.
<img width="374" height="711" alt="image" src="https://github.com/user-attachments/assets/f910d6c3-6d20-4c84-bbc0-903a9a2fe66f" />

#### Made with 🧡 by Simrat




