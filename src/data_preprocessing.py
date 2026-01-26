"""
Amazon Supply Chain Intelligence
Data Preprocessing Pipeline
============================

This module handles all data preprocessing tasks including:
- Data loading and validation
- Missing value imputation
- Outlier detection and handling
- Feature encoding
- Feature scaling
- Target variable creation

Author: ARUN AK
Course: UCS321 - AI for Engineers
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for preprocessing parameters."""

    # Paths
    RAW_DATA_PATH = "data/raw/amazon_delivery.csv"
    PROCESSED_DATA_PATH = "data/processed/"

    # Target variable thresholds (in minutes)
    ONTIME_THRESHOLD = 90
    ATRISK_THRESHOLD = 160

    # Train-test split
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Target classes
    TARGET_CLASSES = {
        0: "On-Time",
        1: "At-Risk",
        2: "Delayed"
    }


# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """Handles data loading and initial validation."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load(self) -> pd.DataFrame:
        """Load the dataset from CSV file."""
        print("=" * 60)
        print("STEP 1: DATA LOADING")
        print("=" * 60)

        self.df = pd.read_csv(self.filepath)

        print(f"✓ Loaded {self.df.shape[0]:,} rows and {self.df.shape[1]} columns")
        print(f"✓ Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        return self.df

    def validate(self) -> dict:
        """Validate data quality and return summary."""
        validation_report = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicates": self.df.duplicated().sum(),
            "dtypes": self.df.dtypes.astype(str).to_dict()
        }

        print(f"✓ Duplicate rows: {validation_report['duplicates']}")
        print(f"✓ Columns with missing values: {sum(1 for v in validation_report['missing_values'].values() if v > 0)}")

        return validation_report


# ============================================================================
# DATA CLEANER
# ============================================================================

class DataCleaner:
    """Handles data cleaning operations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def clean_column_names(self) -> 'DataCleaner':
        """Standardize column names."""
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')
        return self

    def clean_string_columns(self) -> 'DataCleaner':
        """Remove leading/trailing whitespace from string columns."""
        string_cols = self.df.select_dtypes(include=['object']).columns
        for col in string_cols:
            self.df[col] = self.df[col].astype(str).str.strip()
        return self

    def handle_nan_strings(self) -> 'DataCleaner':
        """Convert 'NaN' strings to actual NaN values."""
        self.df = self.df.replace(['NaN', 'nan', 'NULL', 'null', 'None', 'none', ''], np.nan)
        return self

    def drop_duplicates(self) -> 'DataCleaner':
        """Remove duplicate rows."""
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        dropped = initial_rows - len(self.df)
        if dropped > 0:
            print(f"  → Dropped {dropped} duplicate rows")
        return self

    def get_cleaned_data(self) -> pd.DataFrame:
        """Return the cleaned dataframe."""
        return self.df


# ============================================================================
# MISSING VALUE HANDLER
# ============================================================================

class MissingValueHandler:
    """Handles missing value imputation with adaptive strategies."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.imputation_stats = {}

    def impute(self) -> pd.DataFrame:
        """Apply appropriate imputation strategy for each column."""
        print("\n" + "=" * 60)
        print("STEP 2: HANDLING MISSING VALUES")
        print("=" * 60)

        missing_cols = self.df.columns[self.df.isnull().any()].tolist()

        for col in missing_cols:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100

            if self.df[col].dtype in ['int64', 'float64']:
                # Numerical: use median (robust to outliers)
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                self.imputation_stats[col] = {'strategy': 'median', 'value': median_val}
                print(f"  → {col}: Filled {missing_count} ({missing_pct:.2f}%) with median = {median_val:.2f}")
            else:
                # Categorical: use mode
                mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                self.df[col].fillna(mode_val, inplace=True)
                self.imputation_stats[col] = {'strategy': 'mode', 'value': mode_val}
                print(f"  → {col}: Filled {missing_count} ({missing_pct:.2f}%) with mode = '{mode_val}'")

        if not missing_cols:
            print("  ✓ No missing values found!")

        return self.df


# ============================================================================
# OUTLIER HANDLER
# ============================================================================

class OutlierHandler:
    """Detects and handles outliers using IQR method."""

    def __init__(self, df: pd.DataFrame, threshold: float = 1.5):
        self.df = df.copy()
        self.threshold = threshold
        self.outlier_stats = {}

    def detect_outliers(self, column: str) -> tuple:
        """Detect outliers in a numerical column using IQR."""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR

        outliers = ((self.df[column] < lower_bound) | (self.df[column] > upper_bound))

        return outliers, lower_bound, upper_bound

    def handle_outliers(self, columns: list = None, method: str = 'clip') -> pd.DataFrame:
        """Handle outliers using specified method (clip or remove)."""
        print("\n" + "=" * 60)
        print("STEP 3: OUTLIER DETECTION & HANDLING")
        print("=" * 60)

        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        total_outliers = 0

        for col in columns:
            outliers, lower, upper = self.detect_outliers(col)
            outlier_count = outliers.sum()

            if outlier_count > 0:
                total_outliers += outlier_count

                if method == 'clip':
                    self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                    print(f"  → {col}: Clipped {outlier_count} outliers to [{lower:.2f}, {upper:.2f}]")
                elif method == 'remove':
                    self.df = self.df[~outliers]
                    print(f"  → {col}: Removed {outlier_count} outlier rows")

                self.outlier_stats[col] = {
                    'count': outlier_count,
                    'lower_bound': lower,
                    'upper_bound': upper
                }

        if total_outliers == 0:
            print("  ✓ No significant outliers detected!")

        return self.df


# ============================================================================
# FEATURE ENGINEER
# ============================================================================

class FeatureEngineer:
    """Creates new features from existing data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def calculate_distance(self) -> 'FeatureEngineer':
        """Calculate Haversine distance between store and drop location."""
        # Haversine formula for distance calculation
        lat1 = np.radians(self.df['store_latitude'])
        lat2 = np.radians(self.df['drop_latitude'])
        lon1 = np.radians(self.df['store_longitude'])
        lon2 = np.radians(self.df['drop_longitude'])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth's radius in kilometers
        r = 6371

        self.df['distance_km'] = c * r
        return self

    def extract_datetime_features(self) -> 'FeatureEngineer':
        """Extract features from date and time columns."""
        # Parse dates
        self.df['order_date'] = pd.to_datetime(self.df['order_date'], errors='coerce')

        # Extract date features
        self.df['day_of_week'] = self.df['order_date'].dt.dayofweek
        self.df['day_of_month'] = self.df['order_date'].dt.day
        self.df['month'] = self.df['order_date'].dt.month
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)

        # Parse time and extract hour
        self.df['order_hour'] = pd.to_datetime(self.df['order_time'], format='%H:%M:%S', errors='coerce').dt.hour
        self.df['pickup_hour'] = pd.to_datetime(self.df['pickup_time'], format='%H:%M:%S', errors='coerce').dt.hour

        # Time of day categories
        def get_time_period(hour):
            if pd.isna(hour):
                return 'Unknown'
            elif 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Night'

        self.df['time_period'] = self.df['order_hour'].apply(get_time_period)

        # Rush hour flag
        self.df['is_rush_hour'] = self.df['order_hour'].apply(
            lambda x: 1 if (8 <= x <= 10) or (17 <= x <= 19) else 0 if pd.notna(x) else 0
        )

        # Pickup delay (time between order and pickup)
        self.df['pickup_delay_mins'] = (self.df['pickup_hour'] - self.df['order_hour']) * 60
        self.df['pickup_delay_mins'] = self.df['pickup_delay_mins'].clip(lower=0, upper=180)

        return self

    def create_interaction_features(self) -> 'FeatureEngineer':
        """Create interaction features."""
        # Distance per unit time potential
        self.df['distance_traffic_interaction'] = self.df['distance_km'] * self.df['traffic'].map({
            'Low': 1, 'Medium': 2, 'High': 3, 'Jam': 4
        }).fillna(2)

        # Weather severity score
        weather_severity = {
            'Sunny': 1, 'Cloudy': 2, 'Windy': 3,
            'Fog': 4, 'Stormy': 5, 'Sandstorms': 5
        }
        self.df['weather_severity'] = self.df['weather'].map(weather_severity).fillna(2)

        # Combined risk score
        self.df['risk_score'] = (
            self.df['distance_km'] * 0.3 +
            self.df['weather_severity'] * 10 +
            self.df['is_rush_hour'] * 15 +
            self.df['distance_traffic_interaction'] * 0.5
        )

        # Agent experience proxy (higher rating = more experienced)
        self.df['agent_experience_score'] = self.df['agent_rating'] * (1 - (self.df['agent_age'] - 20) / 40)

        return self

    def create_target_variable(self) -> 'FeatureEngineer':
        """Create the target variable based on delivery time thresholds."""
        def categorize_delivery(time):
            if time <= Config.ONTIME_THRESHOLD:
                return 0  # On-Time
            elif time <= Config.ATRISK_THRESHOLD:
                return 1  # At-Risk
            else:
                return 2  # Delayed

        self.df['delivery_status'] = self.df['delivery_time'].apply(categorize_delivery)

        return self

    def get_engineered_data(self) -> pd.DataFrame:
        """Return the dataframe with engineered features."""
        return self.df


# ============================================================================
# FEATURE ENCODER
# ============================================================================

class FeatureEncoder:
    """Handles encoding of categorical variables."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.encoders = {}
        self.encoding_maps = {}

    def encode_categorical(self, columns: list = None) -> pd.DataFrame:
        """Encode categorical columns using Label Encoding."""
        print("\n" + "=" * 60)
        print("STEP 4: FEATURE ENCODING")
        print("=" * 60)

        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns.tolist()
            # Remove ID columns and date columns
            columns = [c for c in columns if not any(x in c.lower() for x in ['id', 'date', 'time'])]

        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                # Handle any remaining NaN
                self.df[col] = self.df[col].fillna('Unknown')
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
                self.encoding_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))
                print(f"  → {col}: Encoded {len(le.classes_)} unique values")

        return self.df


# ============================================================================
# FEATURE SCALER
# ============================================================================

class FeatureScaler:
    """Handles feature scaling."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.scaled_columns = []

    def scale_features(self, columns: list = None) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        print("\n" + "=" * 60)
        print("STEP 5: FEATURE SCALING")
        print("=" * 60)

        if columns is None:
            # Select numerical columns for scaling
            exclude_cols = ['delivery_time', 'delivery_status', 'order_id']
            columns = [c for c in self.df.select_dtypes(include=['int64', 'float64']).columns
                      if c not in exclude_cols and not c.endswith('_encoded')]

        self.scaled_columns = columns

        if columns:
            self.df[columns] = self.scaler.fit_transform(self.df[columns])
            print(f"  ✓ Scaled {len(columns)} numerical features using StandardScaler")

        return self.df


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

class PreprocessingPipeline:
    """Main pipeline orchestrating all preprocessing steps."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = None

    def run(self) -> tuple:
        """Execute the full preprocessing pipeline."""
        print("\n" + "=" * 60)
        print("AMAZON SUPPLY CHAIN INTELLIGENCE")
        print("Data Preprocessing Pipeline")
        print("=" * 60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Load Data
        loader = DataLoader(self.config.RAW_DATA_PATH)
        self.df = loader.load()
        loader.validate()

        # Step 2: Clean Data
        print("\n" + "=" * 60)
        print("STEP 2: DATA CLEANING")
        print("=" * 60)
        cleaner = DataCleaner(self.df)
        self.df = (cleaner
                   .clean_column_names()
                   .clean_string_columns()
                   .handle_nan_strings()
                   .drop_duplicates()
                   .get_cleaned_data())
        print("  ✓ Column names standardized")
        print("  ✓ String columns cleaned")
        print("  ✓ NaN strings converted")

        # Step 3: Handle Missing Values
        mv_handler = MissingValueHandler(self.df)
        self.df = mv_handler.impute()

        # Step 4: Handle Outliers
        outlier_cols = ['agent_age', 'agent_rating', 'delivery_time']
        oh = OutlierHandler(self.df)
        self.df = oh.handle_outliers(columns=outlier_cols, method='clip')

        # Step 5: Feature Engineering
        print("\n" + "=" * 60)
        print("STEP 5: FEATURE ENGINEERING")
        print("=" * 60)
        fe = FeatureEngineer(self.df)
        self.df = (fe
                   .calculate_distance()
                   .extract_datetime_features()
                   .create_interaction_features()
                   .create_target_variable()
                   .get_engineered_data())
        print("  ✓ Distance calculated (Haversine formula)")
        print("  ✓ Datetime features extracted")
        print("  ✓ Interaction features created")
        print("  ✓ Target variable created")

        # Step 6: Encode Categorical Features
        encoder = FeatureEncoder(self.df)
        categorical_cols = ['weather', 'traffic', 'vehicle', 'area', 'category', 'time_period']
        self.df = encoder.encode_categorical(categorical_cols)

        # Step 7: Select Features for Modeling
        print("\n" + "=" * 60)
        print("STEP 6: FEATURE SELECTION")
        print("=" * 60)

        self.feature_columns = [
            'agent_age', 'agent_rating', 'distance_km',
            'day_of_week', 'month', 'is_weekend', 'order_hour',
            'is_rush_hour', 'pickup_delay_mins',
            'weather_severity', 'distance_traffic_interaction',
            'risk_score', 'agent_experience_score',
            'weather_encoded', 'traffic_encoded', 'vehicle_encoded',
            'area_encoded', 'category_encoded', 'time_period_encoded'
        ]

        # Ensure all features exist
        self.feature_columns = [c for c in self.feature_columns if c in self.df.columns]
        print(f"  ✓ Selected {len(self.feature_columns)} features for modeling")

        # Handle any remaining NaN in feature columns
        for col in self.feature_columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(0, inplace=True)

        # Step 8: Split Data
        print("\n" + "=" * 60)
        print("STEP 7: TRAIN-TEST SPLIT")
        print("=" * 60)

        X = self.df[self.feature_columns]
        y = self.df['delivery_status']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )

        print(f"  ✓ Training set: {len(self.X_train):,} samples")
        print(f"  ✓ Test set: {len(self.X_test):,} samples")

        # Print class distribution
        print("\n  Class Distribution:")
        for cls, name in self.config.TARGET_CLASSES.items():
            train_pct = (self.y_train == cls).sum() / len(self.y_train) * 100
            test_pct = (self.y_test == cls).sum() / len(self.y_test) * 100
            print(f"    {name}: Train={train_pct:.1f}%, Test={test_pct:.1f}%")

        # Step 9: Scale Features
        print("\n" + "=" * 60)
        print("STEP 8: FEATURE SCALING")
        print("=" * 60)

        scaler = StandardScaler()
        numerical_cols = [c for c in self.feature_columns if not c.endswith('_encoded')]

        self.X_train[numerical_cols] = scaler.fit_transform(self.X_train[numerical_cols])
        self.X_test[numerical_cols] = scaler.transform(self.X_test[numerical_cols])
        print(f"  ✓ Scaled {len(numerical_cols)} numerical features")

        # Save processed data
        print("\n" + "=" * 60)
        print("STEP 9: SAVING PROCESSED DATA")
        print("=" * 60)

        os.makedirs(self.config.PROCESSED_DATA_PATH, exist_ok=True)

        self.X_train.to_csv(f"{self.config.PROCESSED_DATA_PATH}X_train.csv", index=False)
        self.X_test.to_csv(f"{self.config.PROCESSED_DATA_PATH}X_test.csv", index=False)
        self.y_train.to_csv(f"{self.config.PROCESSED_DATA_PATH}y_train.csv", index=False)
        self.y_test.to_csv(f"{self.config.PROCESSED_DATA_PATH}y_test.csv", index=False)
        self.df.to_csv(f"{self.config.PROCESSED_DATA_PATH}processed_data.csv", index=False)

        print(f"  ✓ Saved processed data to {self.config.PROCESSED_DATA_PATH}")

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return self.X_train, self.X_test, self.y_train, self.y_test


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the preprocessing pipeline
    pipeline = PreprocessingPipeline()
    X_train, X_test, y_train, y_test = pipeline.run()

    print(f"\n✓ X_train shape: {X_train.shape}")
    print(f"✓ X_test shape: {X_test.shape}")
    print(f"✓ y_train shape: {y_train.shape}")
    print(f"✓ y_test shape: {y_test.shape}")
