import sys
import os

# Define the full path to the 'prophet' folder
prophet_path = os.path.join(os.getcwd(), 'myenv', 'Lib', 'site-packages')

# Add the 'prophet' path to sys.path
sys.path.insert(0, prophet_path)

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib
matplotlib.use("Agg")  # Or 'Qt5Agg','TkAgg'
import matplotlib.pyplot as plt
import statsmodels.api as sm
from prophet import Prophet  # Correct import statement
# Function to add date features
def add_date_features(df, date_column_name):
    df["Day"] = df[date_column_name].dt.dayofweek
    df["Month"] = df[date_column_name].dt.month
    df["Year"] = df[date_column_name].dt.year
    df["Q"] = df[date_column_name].dt.quarter
    df["Dayofyear"] = df[date_column_name].dt.dayofyear
    df["Dayofmonth"] = df[date_column_name].dt.day
    df["Weekofyear"] = df[date_column_name].dt.isocalendar().week
    df["Season"] = df["Month"].apply(categorize_season)
    return df

# Function to categorize seasons
def categorize_season(month):
    # Define a mapping of months to seasons
    season_mapping = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 2,
        6: 3,
        7: 3,
        8: 3,
        9: 4,
        10: 4,
        11: 4,
        12: 1
    }

    # Return the corresponding season based on the month
    return season_mapping[month]

# Function to load rainfall data
def load_rainfall_data():
    rainfall_df = pd.read_csv("Kibabii_University_data_Rainfall.csv")
    rainfall_df['ID'] = pd.to_datetime(rainfall_df['ID'], format='%Y%m%d')
    rainfall_df = add_date_features(rainfall_df, 'ID')
    rainfall_df.set_index('ID', inplace=True)
    return rainfall_df

# Function to load temperature max data
def load_temp_max_data():
    temp_max_df = pd.read_csv("Kibabii_University_data_Tmax.csv")
    temp_max_df['ID'] = pd.to_datetime(temp_max_df['ID'], format='%Y%m%d')
    temp_max_df = add_date_features(temp_max_df, 'ID')
    temp_max_df.set_index('ID', inplace=True)
    return temp_max_df

# Function to load temperature min data
def load_temp_min_data():
    temp_min_df = pd.read_csv("Kibabii_University_data_Tmin.csv")
    temp_min_df['ID'] = pd.to_datetime(temp_min_df['ID'], format='%Y%m%d')
    temp_min_df = add_date_features(temp_min_df, 'ID')
    temp_min_df.set_index('ID', inplace=True)
    return temp_min_df

# Plot yearly averages
def plot_yearly_averages(data, column):
    yearly_averages = data.groupby("Year")[column].mean()
    fig = px.line(yearly_averages, x=yearly_averages.index, y=yearly_averages.values, title=f"Yearly Average {column}")
    st.plotly_chart(fig)

# Plot monthly averages
def plot_monthly_averages(data, column):
    monthly_averages = data.groupby("Month")[column].mean()
    # Rename index to month names
    monthly_averages.index = pd.date_range(start='2022-01-01', periods=12, freq='M').month_name()
    fig = px.line(monthly_averages, x=monthly_averages.index, y=monthly_averages.values, title=f"Monthly Average {column}")
    st.plotly_chart(fig)

# Plot seasonal averages
def plot_seasonal_averages(data, column):
    seasonal_averages = data.groupby("Season")[column].mean()
    season_names = {1: "Hot and Dry_Season", 2: "Long Rains", 3: "Dry Season", 4: "Short Rains"}
    seasonal_averages.index = seasonal_averages.index.map(season_names)
    fig = px.bar(seasonal_averages, x=seasonal_averages.index, y=seasonal_averages.values, title=f"Seasonal Average {column}")
    st.plotly_chart(fig)
    
# Plot quarterly averages
def plot_quarterly_averages(data, column, year, quarter):
    quarterly_data = data[(data["Year"] == year) & (data["Q"] == quarter)]
    quarterly_averages = quarterly_data.groupby("Month")[column].mean()
    
    # Determine the correct month labels for the selected quarter
    if quarter == 1:
        month_labels = ["Jan", "Feb", "Mar"]
    elif quarter == 2:
        month_labels = ["Apr", "May", "Jun"]
    elif quarter == 3:
        month_labels = ["Jul", "Aug", "Sep"]
    else:
        month_labels = ["Oct", "Nov", "Dec"]
        
    quarterly_averages.index = month_labels
    
    fig = px.line(quarterly_averages, x=quarterly_averages.index, y=quarterly_averages.values, title=f"Quarterly Average {column} - Q{quarter} {year}")
    st.plotly_chart(fig)

# Function to plot outliers using box plot
def plot_outliers(data, column):
    fig = px.box(data, y=column, title=f"Box Plot - {column}")
    st.plotly_chart(fig)

# Perform seasonal decomposition and plot for each column
def seasonal_decomposition_and_plot(data, column):
    # Perform seasonal decomposition
    s = sm.tsa.seasonal_decompose(data[column], period=365)  # Assuming yearly frequency

    # Create a figure
    fig = plt.figure(figsize=(10, 10))

    # Plot the decomposition
    plt.subplot(4, 1, 1)
    plt.plot(data.index, s.observed)
    plt.title(f'Observed: {column}')
    plt.xlim(pd.Timestamp('1990-01-01'), pd.Timestamp('2022-12-31'))  # Set x-axis limits

    plt.subplot(4, 1, 2)
    plt.plot(data.index, s.trend)
    plt.title(f'Trend: {column}')
    plt.xlim(pd.Timestamp('1990-01-01'), pd.Timestamp('2022-12-31'))  # Set x-axis limits

    plt.subplot(4, 1, 3)
    plt.plot(data.index, s.seasonal)
    plt.title(f'Seasonality: {column}')
    plt.xlim(pd.Timestamp('1990-01-01'), pd.Timestamp('2022-12-31'))  # Set x-axis limits

    plt.subplot(4, 1, 4)
    plt.plot(data.index, s.resid)
    plt.title(f'Residual: {column}')
    plt.xlim(pd.Timestamp('1990-01-01'), pd.Timestamp('2022-12-31'))  # Set x-axis limits
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    
# Page: Home
def page_home():
    st.title("Welcome to Your Weather Data App!")
    st.write("This app allows you to explore rainfall and temperature data.")
    st.write("Please use the sidebar to navigate to different pages.")

    st.header("Research Questions")
    st.markdown("* What are the trends in climatic patterns in North Rift region for the last 30 years?")
    st.markdown("* How do you predict climatic weather patterns?")

    st.header("Data Understanding")
    st.markdown("### Rainfall Dataset")
    st.markdown("The rainfall dataset has data for rainfall data for 8 counties and for a period of 30 years from 1990-2022 and 9 variables:")
    st.markdown("- ID (Date)")
    st.markdown("- TransNzoia")
    st.markdown("- UasinGishu")
    st.markdown("- Nandi")
    st.markdown("- Turkana")
    st.markdown("- Baringo")
    st.markdown("- WestPokot")
    st.markdown("- Samburu")
    st.markdown("- ElgeyoMarakwet")

    st.markdown("### Temperature Datasets")
    st.markdown("The temperature datasets have data for min and max temperature for 8 counties and for a period of 30 years from 1990-2022 and 9 variables:")
    st.markdown("- ID (Date)")
    st.markdown("- TransNzoia")
    st.markdown("- UasinGishu")
    st.markdown("- Nandi")
    st.markdown("- Turkana")
    st.markdown("- Baringo")
    st.markdown("- WestPokot")
    st.markdown("- Samburu")
    st.markdown("- ElgeyoMarakwet")
    
    st.header("Summary")
    st.markdown("""
    The datasets consist of rainfall and temperature data for the North Rift region, comprising 9 variables each and a total of 12053 observations. The datasets have undergone profiling analysis, revealing the following key insights:

    1. **Data Integrity**:
       - There are no missing cells in any of the datasets, indicating that the data is complete and ready for analysis.
       - Additionally, there are no duplicate rows present in the datasets, ensuring data consistency and reliability.

    2. **Variable Description**:
       - Each dataset contains 9 variables, which likely include identifiers such as 'ID' or 'Date' and measurements for different locations or climatic parameters.
       - Rainfall data is reported in millimeters, while temperature data (both maximum and minimum) is recorded in degrees Celsius.

    3. **Dataset Consistency**:
       - All three datasets have the same number of observations, suggesting consistency in data collection and recording processes across variables.

    4. **Data Profiling Reports**:
       - The provided profiling reports offer detailed statistics on the datasets, including the number of variables, observations, missing cells, and duplicate rows.
       - The absence of missing cells and duplicate rows ensures the datasets' quality and reliability for further analysis.

    Based on the profiling reports, the datasets appear to be well-prepared and suitable for comprehensive analysis of climatic patterns in the North Rift region. Further exploration and analysis can be conducted to uncover trends, relationships, and insights within the data, contributing to a deeper understanding of the region's climatic dynamics.

    **Rainfall Columns**:

    * TransNzoia_rain: 5.46% outliers - TransNzoia might experience occasional heavy rainfall due to its geographic location or seasonal weather patterns.
    * UasinGishu_rain: 6.24% outliers - Uasin Gishu might be prone to sudden rainfall variations influenced by local topography or microclimatic factors.
    * Nandi_rain: 3.94% outliers - Nandi may exhibit moderate rainfall outliers caused by local terrain or land use changes.
    * Turkana_rain: 11.32% outliers - Turkana's outliers could be attributed to its arid climate, occasional flash floods, or measurement errors.
    * Baringo_rain: 8.52% outliers - Baringo may experience outliers due to its diverse landscape, including highlands and lowlands, leading to varied rainfall patterns.
    * WestPokot_rain: 8.87% outliers - West Pokot's outliers might result from its rugged terrain and susceptibility to localized weather phenomena.
    * Samburu_rain: 11.39% outliers - Samburu may encounter outliers due to its semi-arid climate, sporadic rainfall, or data collection challenges.
    * ElgeyoMarakwet_rain: 7.50% outliers - Elgeyo Marakwet's outliers could stem from its mountainous terrain or seasonal weather extremes.

    **Min Temperature Columns and Max Temperature Columns**:

    * The outliers in temperature columns could be influenced by various factors such as elevation, proximity to water bodies, urbanization, and land use changes.
    * Turkana, being arid, might have temperature outliers due to extreme daytime heating and rapid cooling at night.
    * Samburu, being semi-arid, might experience temperature outliers due to variations in elevation and vegetation cover affecting local microclimates.
    * Nandi and Uasin Gishu, being highland areas, might have temperature outliers due to altitudinal gradients and topographic features.
    * TransNzoia and Elgeyo Marakwet may exhibit temperature outliers due to their varied topography and altitude differences within the regions.
    * Baringo and West Pokot may have temperature outliers influenced by their diverse landscape, including valleys, plateaus, and hills
    """)
    
    # Seasonality section
    st.header("Seasonality")
    st.write("Seasonal decomposition analysis has been performed for each column separately. You can explore the observed, trend, seasonal, and residual components for each variable.")
    st.write("From the plots of the seasonality component for each region, it's evident that there is noticeable seasonality present in all regions. The seasonality component captures recurring patterns or fluctuations that occur at regular intervals within each region's data. These patterns suggest that in each region follows a distinct seasonal trend, exhibiting periodic variations over time. Understanding and accounting for this seasonality is crucial for accurately modeling and forecasting rainfall patterns in these regions.")
    st.write("You can further explore the seasonal decomposition plots on the each data page.")

    # RMSE section
    st.header("RMSE (Root Mean Squared Error)")
    st.write("The RMSE values for each region obtained from the Prophet model are as follows:")
    
    # Create a table to display RMSE values
    rmse_data = {
        "Region": ["TransNzoia", "UasinGishu", "Nandi", "Turkana", "Baringo", "WestPokot", "Samburu", "ElgeyoMarakwet"],
        "RMSE": [6.43, 7.06, 7.37, 2.38, 4.74, 4.43, 3.10, 6.73]
    }
    rmse_df = pd.DataFrame(rmse_data)
    st.table(rmse_df)

    st.markdown("""
    ### Things to Note:
    - The models used for prediction include ARIMA, SARIMAX, and Prophet, with Prophet performing fairly well in terms of their RMSE values.
    - Given the changes in climatic patterns, the analysis narrowed down to working with a 10-year period for modeling.
    - Outliers were not dropped from the datasets as they were considered important indicators of changes in rainfall across seasons.
    """)


# Page 1: Rainfall Data
def page_rainfall():
    st.title("Rainfall Data in mm")
    rainfall_data = load_rainfall_data()
    show_data = st.checkbox("Show Data")
    if show_data:
        st.write(rainfall_data)
    st.write(rainfall_data.describe())

    # Plotting section
    st.subheader("Plot Rainfall Data")
    selected_column = st.selectbox("Select Column", rainfall_data.columns)
    plot_type = st.selectbox("Select Plot Type", ["Yearly", "Monthly", "Seasonal", "Quarterly", "Outliers","Seasonal Decomposition"])
    if plot_type == "Yearly":
        plot_yearly_averages(rainfall_data, selected_column)
    elif plot_type == "Monthly":
        plot_monthly_averages(rainfall_data, selected_column)
    elif plot_type == "Quarterly":
        selected_year = st.selectbox("Select Year", sorted(rainfall_data["Year"].unique()))
        selected_quarter = st.selectbox("Select Quarter", [1, 2, 3, 4])
        plot_quarterly_averages(rainfall_data, selected_column, selected_year, selected_quarter)
    elif plot_type == "Outliers":
        plot_outliers(rainfall_data, selected_column)
    elif plot_type == "Seasonal Decomposition":
        st.title("Seasonal Decomposition")
        seasonal_decomposition_and_plot(rainfall_data, selected_column)
    else:
        plot_seasonal_averages(rainfall_data, selected_column)


# Page 2: Temperature Max Data
def page_temp_max():
    st.title(u'Temperature Max Data (\u2103)')
    temp_max_data = load_temp_max_data()
    show_data = st.checkbox("Show Data")
    if show_data:
        st.write(temp_max_data)
    st.write(temp_max_data.describe())

    # Plotting section
    st.subheader("Plot Temperature Max Data")
    selected_column = st.selectbox("Select Column", temp_max_data.columns)
    plot_type = st.selectbox("Select Plot Type", ["Yearly", "Monthly", "Seasonal", "Quarterly", "Outliers","Seasonal Decomposition"])
    if plot_type == "Yearly":
        plot_yearly_averages(temp_max_data, selected_column)
    elif plot_type == "Monthly":
        plot_monthly_averages(temp_max_data, selected_column)
    elif plot_type == "Quarterly":
        selected_year = st.selectbox("Select Year", sorted(temp_max_data["Year"].unique()))
        selected_quarter = st.selectbox("Select Quarter", [1, 2, 3, 4])
        plot_quarterly_averages(temp_max_data, selected_column, selected_year, selected_quarter)
    elif plot_type == "Outliers":
        plot_outliers(temp_max_data, selected_column)
    elif plot_type == "Seasonal Decomposition":
        st.title("Seasonal Decomposition")
        seasonal_decomposition_and_plot(temp_max_data, selected_column)
    else:
        plot_seasonal_averages(temp_max_data, selected_column)

# Page 3: Temperature Min Data
def page_temp_min():
    st.title(u"Temperature Min Data (\u2103)")
    temp_min_data = load_temp_min_data()
    show_data = st.checkbox("Show Data")
    if show_data:
        st.write(temp_min_data)
    st.write(temp_min_data.describe())

    # Plotting section
    st.subheader("Plot Temperature Min Data")
    selected_column = st.selectbox("Select Column", temp_min_data.columns)
    plot_type = st.selectbox("Select Plot Type", ["Yearly", "Monthly", "Seasonal", "Quarterly", "Outliers","Seasonal Decomposition"])
    if plot_type == "Yearly":
        plot_yearly_averages(temp_min_data, selected_column)
    elif plot_type == "Monthly":
        plot_monthly_averages(temp_min_data, selected_column)
    elif plot_type == "Quarterly":
        selected_year = st.selectbox("Select Year", sorted(temp_min_data["Year"].unique()))
        selected_quarter = st.selectbox("Select Quarter", [1, 2, 3, 4])
        plot_quarterly_averages(temp_min_data, selected_column, selected_year, selected_quarter)
    elif plot_type == "Outliers":
        plot_outliers(temp_min_data, selected_column)
    elif plot_type == "Seasonal Decomposition":
        st.title("Seasonal Decomposition")
        seasonal_decomposition_and_plot(temp_min_data, selected_column)
    else:
        plot_seasonal_averages(temp_min_data, selected_column)

TransNzoia = pd.read_csv('TransNzoia.csv')
UasinGishu = pd.read_csv('UasinGishu.csv')
Nandi = pd.read_csv('Nandi.csv')
Turkana = pd.read_csv('Turkana.csv')
Baringo = pd.read_csv('Baringo.csv')
WestPokot = pd.read_csv('WestPokot.csv')
Samburu = pd.read_csv('Samburu.csv')
ElgeyoMarakwet = pd.read_csv('ElgeyoMarakwet.csv')

def prophet_predictions(df, date_input):
    # Rename columns to 'ds' and 'y' for Prophet
    df.rename(columns={'ID': 'ds', df.columns[1]: 'y'}, inplace=True)

    # Create and fit Prophet model
    model = Prophet()
    model.fit(df)

    # Make future dataframe for prediction
    future = pd.DataFrame({'ds': [date_input]})

    # Predict
    forecast = model.predict(future)

    return forecast

def page_prophet_predictions():
    st.title("Prophet Predictions")

    # Select dataframe
    selected_df = st.selectbox("Select DataFrame", ["TransNzoia", "UasinGishu", "Nandi", "Turkana", "Baringo", "WestPokot", "Samburu", "ElgeyoMarakwet"])

    # Select date
    date_input = st.date_input("Select Date for Prediction")

    # Perform Prophet predictions
    if st.button("Predict"):
        if selected_df == "TransNzoia":
            forecast = prophet_predictions(TransNzoia, date_input)
        elif selected_df == "UasinGishu":
            forecast = prophet_predictions(UasinGishu, date_input)
        elif selected_df == "Nandi":
            forecast = prophet_predictions(Nandi, date_input)
        elif selected_df == "Turkana":
            forecast = prophet_predictions(Turkana, date_input)
        elif selected_df == "Baringo":
            forecast = prophet_predictions(Baringo, date_input)
        elif selected_df == "WestPokot":
            forecast = prophet_predictions(WestPokot, date_input)
        elif selected_df == "Samburu":
            forecast = prophet_predictions(Samburu, date_input)
        elif selected_df == "ElgeyoMarakwet":
            forecast = prophet_predictions(ElgeyoMarakwet, date_input)

        # Display forecast
        st.subheader("Prophet Forecast:")
        st.write(f"Predicted Value (yhat): {forecast['yhat'].values[0]}")
        st.write(f"Lower Bound (ylower): {forecast['yhat_lower'].values[0]}")
        st.write(f"Upper Bound (yupper): {forecast['yhat_upper'].values[0]}")


# Main function to run the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Home", "Rainfall Data", "Temperature Max Data", "Temperature Min Data", "Prophet Predictions"])

    if page == "Home":
        page_home()
    elif page == "Rainfall Data":
        page_rainfall()
    elif page == "Temperature Max Data":
        page_temp_max()
    elif page == "Temperature Min Data":
        page_temp_min()
    elif page == "Prophet Predictions":
        page_prophet_predictions()


if __name__ == "__main__":
    main()
