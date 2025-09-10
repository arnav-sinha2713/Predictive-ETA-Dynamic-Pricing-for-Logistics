import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

# Configure visualization styles
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
warnings.filterwarnings('ignore')

file_path = "/content/drive/MyDrive/Colab Notebooks/Project 1/train.csv"

try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"--- ERROR: File Not Found ---")
    print(f"Please upload '{file_path}' to your Colab session.")

# Display basic information
if 'df' in locals():
    print("\nInitial data shape:", df.shape)
    print("\nFirst 5 rows of the dataset:")
    display(df.head())
    print("\nData information and types:")
    df.info()
    print("\nDescriptive statistics for numerical columns:")
    display(df.describe())

def clean_data(df):
    """Cleans and preprocesses the raw dataframe."""
    print("\nCleaning data...")

    # Convert string 'NaN' in time columns to actual NaN values before dropping
    df['Time_Orderd'] = df['Time_Orderd'].apply(lambda x: np.nan if str(x).strip() == 'NaN' else x)
    df['Time_Order_picked'] = df['Time_Order_picked'].apply(lambda x: np.nan if str(x).strip() == 'NaN' else x)

    # Drop rows with missing crucial information
    df.dropna(subset=['Delivery_person_Age', 'Delivery_person_Ratings', 'Time_Orderd', 'Time_Order_picked'], inplace=True)

    # Clean up string NaN values and strip whitespace from categorical columns
    for col in ['Weatherconditions', 'Road_traffic_density', 'Type_of_vehicle', 'Festival', 'City']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.nan if str(x).strip() == 'NaN' else x)
            df[col] = df[col].str.strip()

    # Remove "conditions " prefix from Weatherconditions
    df['Weatherconditions'] = df['Weatherconditions'].str.replace('conditions ', '')

    # Clean and convert target variable 'Time_taken(min)'
    df['Time_taken(min)'] = pd.to_numeric(df['Time_taken(min)'].str.replace('(min) ', ''), errors='coerce')

    # Convert other columns to appropriate numeric types
    df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
    df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')

    # Handle datetime conversion
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y')
    df['Time_Orderd'] = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Time_Orderd'])
    df['Time_Order_picked'] = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Time_Order_picked'])

    # Drop any remaining rows that became NaN during conversion
    df.dropna(inplace=True)

    print("Data cleaning complete.")
    print("Data shape after cleaning:", df.shape)
    return df

df_cleaned = clean_data(df.copy())

plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Time_taken(min)'], kde=True, bins=30)
plt.title('Distribution of Delivery Time (in minutes)')
plt.xlabel('Time Taken (min)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Road_traffic_density', y='Time_taken(min)', data=df_cleaned, order=['Low', 'Medium', 'High', 'Jam'])
plt.title('Delivery Time vs. Road Traffic Density')
plt.xlabel('Road Traffic Density')
plt.ylabel('Time Taken (min)')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18, 7))

sns.boxplot(x='Weatherconditions', y='Time_taken(min)', data=df_cleaned, ax=ax[0])
ax[0].set_title('Delivery Time vs. Weather Conditions')
ax[0].tick_params(axis='x', rotation=45)

sns.boxplot(x='City', y='Time_taken(min)', data=df_cleaned, ax=ax[1])
ax[1].set_title('Delivery Time vs. City Type')

plt.tight_layout()
plt.show()

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in km
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def feature_engineer(df):
    print("\nPerforming feature engineering...")

    # Calculate delivery distance in km
    df['distance_km'] = haversine_distance(df['Restaurant_latitude'], df['Restaurant_longitude'],
                                           df['Delivery_location_latitude'], df['Delivery_location_longitude'])

    # Calculate restaurant preparation time in minutes
    df['preparation_time_mins'] = (df['Time_Order_picked'] - df['Time_Orderd']).dt.total_seconds() / 60

    # Extract time-based features
    df['order_hour'] = df['Time_Orderd'].dt.hour
    df['day_of_week'] = df['Time_Orderd'].dt.dayofweek

    # Drop original columns that are no longer needed
    df_engineered = df.drop([
        'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude',
        'Delivery_location_longitude', 'ID', 'Delivery_person_ID', 'Order_Date',
        'Time_Orderd', 'Time_Order_picked'
    ], axis=1)

    print("Feature engineering complete.")
    return df_engineered

df_engineered = feature_engineer(df_cleaned.copy())

# Let's visualize the new distance feature against our target
plt.figure(figsize=(10, 6))
sns.scatterplot(x='distance_km', y='Time_taken(min)', data=df_engineered, alpha=0.5)
plt.title('Delivery Time vs. Distance')
plt.xlabel('Distance (km)')
plt.ylabel('Time Taken (min)')
plt.show()


def train_model(df):
    print("\nTraining ETA prediction model...")

    X = df.drop('Time_taken(min)', axis=1)
    y = df['Time_taken(min)']

    # One-Hot Encode categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_split=10)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model training complete.")
    print(f"Model Mean Absolute Error (MAE): {mae:.2f} minutes")
    print(f"Model R-squared (R2): {r2:.2f}")

    return model, X_encoded.columns, X_train, y_train

eta_model, model_cols, X_train_cols, y_train_data = train_model(df_engineered)

importances = pd.Series(eta_model.feature_importances_, index=X_train_cols.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 7))
sns.barplot(x=importances, y=importances.index)
plt.title('Top 10 Most Important Features')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

def calculate_delivery_price(order_details, model, model_columns):
    # Pricing Structure & Surcharges
    BASE_FARE = 40.0; PER_KM_RATE = 8.0; PER_MINUTE_RATE = 1.0
    TRAFFIC_SURCHARGE = {'Jam': 25.0, 'High': 15.0, 'Medium': 5.0, 'Low': 0.0}
    WEATHER_SURCHARGE = {'Stormy': 20.0, 'Windy': 15.0, 'Fog': 20.0, 'Sandstorms': 25.0, 'Sunny': 0.0, 'Cloudy': 0.0}
    PEAK_HOUR_SURCHARGE = 10.0

    # 1. Predict ETA
    order_df = pd.DataFrame([order_details])
    order_encoded = pd.get_dummies(order_df).reindex(columns=model_columns, fill_value=0)
    predicted_eta = model.predict(order_encoded)[0]

    # 2. Calculate Price Components
    price = BASE_FARE
    price += order_details['distance_km'] * PER_KM_RATE
    price += predicted_eta * PER_MINUTE_RATE
    price += TRAFFIC_SURCHARGE.get(order_details['Road_traffic_density'], 0.0)
    price += WEATHER_SURCHARGE.get(order_details['Weatherconditions'], 0.0)
    if order_details['order_hour'] in [12, 13, 14, 19, 20, 21]:
        price += PEAK_HOUR_SURCHARGE

    return round(predicted_eta, 2), round(price, 2)

# --- Example Usage ---
print("\n--- Testing Dynamic Pricing ---")
# Example 1: Easy order
sample_order_1 = {
    'Delivery_person_Age': 25, 'Delivery_person_Ratings': 4.8, 'Weatherconditions': 'Sunny',
    'Road_traffic_density': 'Low', 'Vehicle_condition': 2, 'Type_of_vehicle': 'scooter',
    'multiple_deliveries': 1.0, 'Festival': 'No', 'City': 'Urban', 'distance_km': 3.5,
    'preparation_time_mins': 15.0, 'order_hour': 15, 'day_of_week': 2
}
eta_1, price_1 = calculate_delivery_price(sample_order_1, eta_model, model_cols)
print(f"Order 1 (Easy): Predicted ETA = {eta_1} mins, Dynamic Price = ₹{price_1}")

# Example 2: Difficult order
sample_order_2 = {
    'Delivery_person_Age': 35, 'Delivery_person_Ratings': 4.5, 'Weatherconditions': 'Stormy',
    'Road_traffic_density': 'Jam', 'Vehicle_condition': 1, 'Type_of_vehicle': 'motorcycle',
    'multiple_deliveries': 1.0, 'Festival': 'No', 'City': 'Metropolitian', 'distance_km': 12.0,
    'preparation_time_mins': 25.0, 'order_hour': 19, 'day_of_week': 4
}
eta_2, price_2 = calculate_delivery_price(sample_order_2, eta_model, model_cols)
print(f"Order 2 (Hard): Predicted ETA = {eta_2} mins, Dynamic Price = ₹{price_2}")