"""Turns raw form inputs into the same features the model was trained on.

Mirrors src/data_preprocessing.py.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [
    'agent_age', 'agent_rating', 'distance_km',
    'day_of_week', 'month', 'is_weekend', 'order_hour',
    'is_rush_hour', 'pickup_delay_mins',
    'weather_severity', 'distance_traffic_interaction',
    'risk_score', 'agent_experience_score',
    'weather_encoded', 'traffic_encoded', 'vehicle_encoded',
    'area_encoded', 'category_encoded', 'time_period_encoded'
]
SCALED = [c for c in FEATURES if not c.endswith('_encoded')]
CATEGORICAL = ['weather', 'traffic', 'vehicle', 'area', 'category', 'time_period']
WEATHER_SEVERITY = {'Sunny': 1, 'Cloudy': 2, 'Windy': 3, 'Fog': 4, 'Stormy': 5, 'Sandstorms': 5}
TRAFFIC_LEVEL = {'Low': 1, 'Medium': 2, 'High': 3, 'Jam': 4}


def fit_preprocessor(df):
    """Rebuild category codes and the scaler from processed_data.csv (unscaled)."""
    codes = {}
    for col in CATEGORICAL:
        pairs = df[[col, f'{col}_encoded']].dropna().drop_duplicates()
        codes[col] = {str(k): int(v) for k, v in pairs.itertuples(index=False)}
    # Same split settings as the training pipeline
    train, _ = train_test_split(df[FEATURES], test_size=0.2, random_state=42,
                                stratify=df['delivery_status'])
    return codes, StandardScaler().fit(train[SCALED])


def time_period(hour):
    if 6 <= hour < 12:
        return 'Morning'
    if 12 <= hour < 17:
        return 'Afternoon'
    if 17 <= hour < 21:
        return 'Evening'
    return 'Night'


def raw_features(codes, agent_age, agent_rating, distance_km, day_of_week, month,
                 order_hour, pickup_delay_mins, weather, traffic, vehicle, area, category):
    is_rush_hour = int(8 <= order_hour <= 10 or 17 <= order_hour <= 19)
    severity = WEATHER_SEVERITY.get(weather, 2)
    interaction = distance_km * TRAFFIC_LEVEL.get(traffic, 2)
    values = {
        'agent_age': agent_age,
        'agent_rating': agent_rating,
        'distance_km': distance_km,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': int(day_of_week in (5, 6)),
        'order_hour': order_hour,
        'is_rush_hour': is_rush_hour,
        'pickup_delay_mins': pickup_delay_mins,
        'weather_severity': severity,
        'distance_traffic_interaction': interaction,
        'risk_score': distance_km * 0.3 + severity * 10 + is_rush_hour * 15 + interaction * 0.5,
        'agent_experience_score': agent_rating * (1 - (agent_age - 20) / 40),
    }
    chosen = dict(weather=weather, traffic=traffic, vehicle=vehicle, area=area,
                  category=category, time_period=time_period(order_hour))
    for col, value in chosen.items():
        values[f'{col}_encoded'] = codes[col][value]  # unknown values raise an error
    return values


def to_model_input(rows, scaler):
    frame = pd.DataFrame(rows)[FEATURES].astype(float)
    frame[SCALED] = scaler.transform(frame[SCALED])
    return frame
