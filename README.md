# SafeCity:Advanced Crime Data Analysis and Visualization

## Overview
This project is designed to provide advanced crime data analysis and visualization tools. It includes features such as hotspot identification, time-based analysis, crime type analysis, predictive modeling, and comparative analysis. The application is built using Python with libraries like Streamlit, Pandas, GeoPandas, Scikit-learn, Folium, and more.

## Features
- **Hotspot Analysis:** Identifies and visualizes crime hotspots using K-Means clustering.
- **Time-based Analysis:** Analyzes crime data based on time units like hour, day of the week, and month.
- **Crime Type Analysis:** Displays the most frequent types of crimes and their percentages.
- **Predictive Modeling:** Uses a Random Forest Regressor to predict future crime rates for the next 30 days.
- **Comparative Analysis:** Compares crime data across different years to identify trends and patterns.

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/crime-data-analysis.git
    cd crime-data-analysis
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment:**
   - On Windows:
     ```sh
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

4. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

5. **Run the application:**
    ```sh
    streamlit run app.py
    ```

## Usage
1. Upload a CSV file with crime data, or use the default file provided in the code.
2. Use the sidebar to filter data by crime type and date range.
3. Select an analysis type from the sidebar to visualize the data accordingly.

## File Structure

- **app.py:** The main script for the Streamlit app.
- **README.md:** Project documentation.
- **LICENSE:** License for the project.

## Requirements

- Python 3.7+
- Pandas
- GeoPandas
- Scikit-learn
- Folium
- Streamlit
- Matplotlib
- Seaborn

## Sample Data
The application is designed to work with a crime data CSV file that includes the following columns:
- `YEAR`
- `MONTH`
- `DAY`
- `HOUR`
- `MINUTE`
- `X` (Longitude)
- `Y` (Latitude)
- `TYPE` (Type of crime)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to all the developers of the open-source libraries used in this project.

