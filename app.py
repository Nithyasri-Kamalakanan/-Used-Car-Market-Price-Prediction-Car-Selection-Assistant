import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="Used Car Assistant", page_icon="🚗", layout="wide")

st.title("🚗 Used Car Market Price Prediction & Car Selection Assistant")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('malaysia_used_cars.csv')
        
        # Clean column names - remove spaces and standardize
        df.columns = df.columns.str.strip().str.replace(' ', '_')
                
        # Extract year and ensure Model is present from CSV
        df['Year'] = df['Description'].str.extract(r'(\d{4})')[0]
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        # use the `Model` column from the CSV (keep as string). Fill missing with a friendly label.
        df['Model'] = df['Model'].fillna('Not available').astype(str)
        df['Model'] = df['Model'].replace('nan', 'Not available')
        df['Car_Age'] = 2024 - df['Year']
       
        # Clean List_Price: remove currency text, commas, and non-numeric parts
        df['List_Price'] = df['List_Price'].astype(str).str.replace('\n', ' ', regex=True)
        price_extract = df['List_Price'].str.extract(r'(\d[\d,]*\.?\d*)')[0]
        df['List_Price'] = pd.to_numeric(price_extract.str.replace(',', ''), errors='coerce')

        # Clean Milleage (note CSV has column "Milleage"): extract first number and handle 'K' (thousands)
        df['Milleage'] = df['Milleage'].astype(str)
        mile_num = df['Milleage'].str.extract(r'(\d+(?:\.\d+)?)')[0]
        mile_num = pd.to_numeric(mile_num, errors='coerce')
        has_k = df['Milleage'].str.contains('[Kk]', na=False)
        df['Milleage'] = mile_num.where(~has_k, mile_num * 1000)

        # Drop rows with missing or non-numeric essential fields
        df = df.dropna(subset=['List_Price', 'Milleage', 'Year'])

        # Replace missing textual fields (or literal 'nan' strings) with a clear label
        cols_to_fix = [c for c in ['Model', 'Gear_Type', 'Location', 'Description'] if c in df.columns]
        if cols_to_fix:
            df[cols_to_fix] = df[cols_to_fix].fillna('Not available')
            df[cols_to_fix] = df[cols_to_fix].replace('nan', 'Not available')

        # Remove invalid data
        df = df[(df['List_Price'] > 0) & (df['Milleage'] >= 0) & (df['Year'] >= 1990)]
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Train model
@st.cache_resource
def train_model(df):
    try:
        # Encode categorical features
        le_model = LabelEncoder()
        le_gear = LabelEncoder()
        
        df_model = df.copy()
        df_model['Model_Encoded'] = le_model.fit_transform(df_model['Model'].astype(str))
        df_model['Gear_Encoded'] = le_gear.fit_transform(df_model['Gear_Type'].astype(str))
        
        # Prepare features
        X = df_model[['Year', 'Milleage', 'Car_Age', 'Model_Encoded', 'Gear_Encoded']]
        y = df_model['List_Price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
        model.fit(X_train, y_train)
        
        # Calculate score
        score = model.score(X_test, y_test)
        
        return model, le_model, le_gear, score
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, 0

# Load data
df = load_data()

if df is not None and len(df) > 0:
    model, le_model, le_gear, score = train_model(df)
    
    if model is not None:
        st.sidebar.success(f"✅ Model trained! R² Score: {score:.3f}")
        
        # Sidebar navigation
        page = st.sidebar.radio("Navigation", ["📊 Overview", "💰 Price Prediction", "🎯 Recommendations"])
        
        # PAGE 1: Overview
        if page == "📊 Overview":
            st.header("Market Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Cars", f"{len(df):,}")
            col2.metric("Avg Price", f"RM {df['List_Price'].mean():,.0f}")
            col3.metric("Median Price", f"RM {df['List_Price'].median():,.0f}")
            col4.metric("Avg Mileage", f"{df['Milleage'].mean():,.0f} km")
            
            st.subheader("Top 10 Models")
            model_counts = df['Model'].value_counts().head(10)
            st.bar_chart(model_counts)
            
            st.subheader("Price Distribution")
            st.line_chart(df['List_Price'].value_counts().sort_index())
        
        # PAGE 2: Price Prediction
        elif page == "💰 Price Prediction":
            st.header("Predict Car Price")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_model = st.selectbox("Model", sorted(df['Model'].unique()))
                year = st.number_input("Year", 1990, 2024, 2020)
                mileage = st.number_input("Mileage (km)", 0, 500000, 50000, step=5000)
            with col2:
                gear_type = st.selectbox("Gear Type", sorted(df['Gear_Type'].unique()))
            
            if st.button("🔮 Predict Price", type="primary"):
                try:
                    car_age = 2024 - year
                    model_encoded = le_model.transform([selected_model])[0]
                    gear_encoded = le_gear.transform([gear_type])[0]
                    
                    input_data = pd.DataFrame({
                        'Year': [year],
                        'Milleage': [mileage],
                        'Car_Age': [car_age],
                        'Model_Encoded': [model_encoded],
                        'Gear_Encoded': [gear_encoded]
                    })
                    
                    predicted_price = model.predict(input_data)[0]
                    
                    st.success("✅ Prediction Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Predicted Price", f"RM {predicted_price:,.0f}")
                    
                    # Find similar cars
                    similar = df[(df['Model'] == selected_model) & (abs(df['Year'] - year) <= 2)]
                    if len(similar) > 0:
                        avg_market = similar['List_Price'].mean()
                        col2.metric("Market Average", f"RM {avg_market:,.0f}")
                        diff = predicted_price - avg_market
                        col3.metric("Difference", f"RM {diff:,.0f}", f"{(diff/avg_market)*100:.1f}%")
                    
                    st.info(f"💡 Fair price range: RM {predicted_price*0.9:,.0f} - RM {predicted_price*1.1:,.0f}")
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        
        # PAGE 3: Recommendations
        elif page == "🎯 Recommendations":
            st.header("Find Your Perfect Car")
            
            col1, col2 = st.columns(2)
            with col1:
                budget_min = st.number_input("Min Budget (RM)", 0, 500000, 30000, 5000)
                budget_max = st.number_input("Max Budget (RM)", 0, 1000000, 100000, 5000)
                max_mileage = st.number_input("Max Mileage (km)", 0, 500000, 100000, 10000)
            with col2:
                models = st.multiselect("Preferred Models (optional)", sorted(df['Model'].unique()))
                min_year = st.number_input("Min Year", 1990, 2024, 2015)
            
            if st.button("🔍 Find Cars", type="primary"):
                try:
                    filtered = df[
                        (df['List_Price'] >= budget_min) &
                        (df['List_Price'] <= budget_max) &
                        (df['Milleage'] <= max_mileage) &
                        (df['Year'] >= min_year)
                    ].copy()
                    
                    if models:
                        filtered = filtered[filtered['Model'].isin(models)]
                    
                    if len(filtered) == 0:
                        st.warning("⚠️ No cars found. Try adjusting your filters.")
                    else:
                        # Calculate match score
                        filtered['price_score'] = 1 - (filtered['List_Price'] - budget_min) / (budget_max - budget_min)
                        filtered['mileage_score'] = 1 - (filtered['Milleage'] / max_mileage)
                        filtered['age_score'] = (filtered['Year'] - min_year) / (2024 - min_year)
                        filtered['match_score'] = (
                            filtered['price_score'] * 0.3 +
                            filtered['mileage_score'] * 0.4 +
                            filtered['age_score'] * 0.3
                        ) * 100
                        
                        top_cars = filtered.nlargest(10, 'match_score')
                        
                        st.success(f"✅ Found {len(filtered)} cars. Showing top 10 recommendations:")
                        
                        for idx, car in top_cars.iterrows():
                            with st.expander(f"**{car['Description']}** - RM {car['List_Price']:,.0f} | Match: {car['match_score']:.0f}%"):
                                col1, col2, col3 = st.columns(3)
                                col1.write(f"📍 **Location:** {car['Location']}")
                                col2.write(f"🛣️ **Mileage:** {car['Milleage']:,.0f} km")
                                col3.write(f"⚙️ **Gear:** {car['Gear_Type']}")
                                col1.write(f"📅 **Year:** {int(car['Year'])}")
                                col2.write(f"🏷️ **Model:** {car['Model']}")
                
                except Exception as e:
                    st.error(f"Error finding cars: {str(e)}")