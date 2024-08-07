import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import folium
from folium.plugins import HeatMap, MarkerCluster, Draw
import streamlit as st
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

# Function to load and preprocess data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.upper()
    df['DATETIME'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE']])
    return df.dropna(subset=['X', 'Y'])

# Function to identify hotspots
def identify_hotspots(gdf, n_clusters=5):
    coords = gdf[['X', 'Y']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    gdf['CLUSTER'] = kmeans.fit_predict(coords)
    
    hotspots = []
    for i, center in enumerate(kmeans.cluster_centers_):
        cluster_points = gdf[gdf['CLUSTER'] == i]
        hotspots.append({
            'location': center,
            'count': len(cluster_points),
            'main_crime_type': cluster_points['TYPE'].mode().iloc[0]
        })
    return sorted(hotspots, key=lambda x: x['count'], reverse=True)

# Function for predictive modeling
def predict_crime_rates(df):
    df['DATE'] = df['DATETIME'].dt.date
    daily_crimes = df.groupby('DATE').size().reset_index(name='CRIME_COUNT')
    daily_crimes['DATE'] = pd.to_datetime(daily_crimes['DATE'])
    daily_crimes = daily_crimes.set_index('DATE')
    daily_crimes['DAY_OF_WEEK'] = daily_crimes.index.dayofweek
    daily_crimes['MONTH'] = daily_crimes.index.month
    
    X = daily_crimes[['DAY_OF_WEEK', 'MONTH']]
    y = daily_crimes['CRIME_COUNT']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    future_dates = pd.date_range(start=daily_crimes.index[-1] + timedelta(days=1), periods=30)
    future_X = pd.DataFrame({'DAY_OF_WEEK': future_dates.dayofweek, 'MONTH': future_dates.month})
    future_predictions = model.predict(future_X)
    
    return pd.Series(future_predictions, index=future_dates)

# Function for filtering data
def filter_data(df):
    st.sidebar.subheader("Filter Options")
    crime_types = st.sidebar.multiselect("Select Crime Types", df['TYPE'].unique(), default=df['TYPE'].unique())
    date_range = st.sidebar.date_input("Select Date Range", [df['DATETIME'].min(), df['DATETIME'].max()])
    
    filtered_df = df[(df['TYPE'].isin(crime_types)) & (df['DATETIME'].dt.date.between(*date_range))]
    return filtered_df

# Streamlit app
# Streamlit app
def main():
    st.title("SafeCity: An Advanced Crime Data Analysis and Visualization")

    # Default file path
    default_file_path = 'C:/Users/soumy/LLM/centemental analysis/crimedata_csv_all_years.csv'
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # If no file is uploaded, use the default file
    if uploaded_file is None:
        file = default_file_path
    else:
        file = uploaded_file

    df = load_data(file)

    required_columns = ['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'X', 'Y', 'TYPE']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
        return

    # Apply filters
    df = filter_data(df)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y))

    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.selectbox("Choose Analysis Type", 
                                         ["Hotspot Analysis", "Time-based Analysis", "Crime Type Analysis",
                                          "Predictive Modeling", "Comparative Analysis"])

    if analysis_type == "Hotspot Analysis":
        st.header("Crime Hotspot Analysis")
        
        n_clusters = st.slider("Number of Hotspots", 3, 20, 5)
        hotspots = identify_hotspots(gdf, n_clusters)

        m = folium.Map(location=[gdf.Y.mean(), gdf.X.mean()], zoom_start=12)
        
        # Heat map
        heat_data = [[row['Y'], row['X']] for _, row in gdf.iterrows()]
        HeatMap(heat_data).add_to(m)

        # Cluster markers
        marker_cluster = MarkerCluster().add_to(m)
        for i, hotspot in enumerate(hotspots, 1):
            folium.Marker(
                location=[hotspot['location'][1], hotspot['location'][0]],
                popup=f"Hotspot {i}<br>Crimes: {hotspot['count']}<br>Main Type: {hotspot['main_crime_type']}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(marker_cluster)

        # Add draw control for user-defined areas
        draw = Draw(
            draw_options={'polyline': False, 'rectangle': True, 'polygon': True, 'circle': False, 'marker': False, 'circlemarker': False},
            edit_options={'featureGroup': None})
        draw.add_to(m)

        folium_static(m)

        st.subheader("Top Crime Hotspots")
        for i, hotspot in enumerate(hotspots[:5], 1):
            st.write(f"{i}. Location: ({hotspot['location'][1]:.4f}, {hotspot['location'][0]:.4f}), "
                     f"Crimes: {hotspot['count']}, Main Type: {hotspot['main_crime_type']}")

    elif analysis_type == "Time-based Analysis":
        st.header("Time-based Crime Analysis")

        time_unit = st.selectbox("Select Time Unit", ["Hour", "Day of Week", "Month"])
        
        if time_unit == "Hour":
            df['Hour'] = df['DATETIME'].dt.hour
            time_data = df.groupby('Hour').size()
            x_label = "Hour of Day"
        elif time_unit == "Day of Week":
            df['Day'] = df['DATETIME'].dt.dayofweek
            time_data = df.groupby('Day').size()
            x_label = "Day of Week"
        else:  # Month
            df['Month'] = df['DATETIME'].dt.month
            time_data = df.groupby('Month').size()
            x_label = "Month"

        fig, ax = plt.subplots(figsize=(12, 6))
        time_data.plot(kind='bar', ax=ax)
        plt.title(f"Crime Frequency by {time_unit}")
        plt.xlabel(x_label)
        plt.ylabel("Number of Crimes")
        st.pyplot(fig)

    elif analysis_type == "Crime Type Analysis":
        st.header("Crime Type Analysis")

        top_crimes = df['TYPE'].value_counts().nlargest(10)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        top_crimes.plot(kind='bar', ax=ax)
        plt.title("Top 10 Crime Types")
        plt.xlabel("Crime Type")
        plt.ylabel("Number of Occurrences")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

        st.subheader("Percentage of Top Crime Types")
        st.write(top_crimes.apply(lambda x: f"{x / len(df) * 100:.2f}%"))

    elif analysis_type == "Predictive Modeling":
        st.header("Crime Rate Prediction")
        
        predictions = predict_crime_rates(df)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        predictions.plot(ax=ax)
        plt.title("Predicted Daily Crime Counts for the Next 30 Days")
        plt.xlabel("Date")
        plt.ylabel("Predicted Number of Crimes")
        st.pyplot(fig)

    elif analysis_type == "Comparative Analysis":
        st.header("Comparative Crime Analysis")
        
        years = sorted(df['YEAR'].unique())
        selected_years = st.multiselect("Select years to compare", years, default=years[:2])
        
        if len(selected_years) > 1:
            comparison_data = df[df['YEAR'].isin(selected_years)]
            crime_counts = comparison_data.groupby(['YEAR', 'TYPE']).size().unstack(fill_value=0)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            crime_counts.plot(kind='bar', ax=ax)
            plt.title("Crime Type Comparison Across Years")
            plt.xlabel("Year")
            plt.ylabel("Number of Crimes")
            plt.legend(title="Crime Type", bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)
        else:
            st.warning("Please select more than one year for comparison.")

if __name__ == "__main__":
    main()
