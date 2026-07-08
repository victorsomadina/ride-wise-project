import json
from pathlib import Path
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

BASE_DIR = Path(__file__).resolve().parent
TRIPS_PATH = BASE_DIR / '../data/trips.csv'
RIDERS_PATH = BASE_DIR / '../data/riders.csv'
SESSIONS_PATH = BASE_DIR / '../data/sessions.csv'
OUTPUT_PATH = BASE_DIR / '../data/model_input.parquet'
FEATURE_COLUMNS_PATH = BASE_DIR / '../models/feature_columns.json'

RECENT_WINDOW_DAYS = 30
PRIOR_WINDOW_DAYS = 30


def load_data():
    trips = pd.read_csv(TRIPS_PATH, parse_dates=['pickup_time', 'dropoff_time'])
    riders = pd.read_csv(RIDERS_PATH, parse_dates=['signup_date'])
    sessions = pd.read_csv(SESSIONS_PATH, parse_dates=['session_time'])
    return trips, riders, sessions


def build_trip_features(trips, cutoff=None):
    trips = trips.copy()
    trips['pickup_time'] = pd.to_datetime(trips['pickup_time'], utc=True)

    if cutoff is None:
        cutoff = trips['pickup_time'].max().normalize() + pd.Timedelta(days=1)

    recent_start = cutoff - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    prior_start = recent_start - pd.Timedelta(days=PRIOR_WINDOW_DAYS)

    trip_agg = trips.groupby('user_id').agg(
        total_trips=('trip_id', 'count'),
        total_spent=('fare', 'sum'),
        total_tip=('tip', 'sum'),
        avg_surge=('surge_multiplier', 'mean'),
        last_trip=('pickup_time', 'max'),
    )

    recent_trips = (
        trips[trips['pickup_time'] >= recent_start]
        .groupby('user_id').size().rename('recent_trips')
    )
    prior_trips = (
        trips[(trips['pickup_time'] >= prior_start) & (trips['pickup_time'] < recent_start)]
        .groupby('user_id').size().rename('prior_trips')
    )

    feats = trip_agg.join(recent_trips).join(prior_trips)
    for col in ['recent_trips', 'prior_trips']:
        feats[col] = feats[col].fillna(0)
    feats['trip_trend'] = feats['recent_trips'] - feats['prior_trips']
    feats['recency_at_cutoff'] = (cutoff - feats['last_trip']).dt.days

    return feats.drop(columns='last_trip'), cutoff


def build_session_features(sessions):
    sessions = sessions.rename(columns={'rider_id': 'user_id'})
    return sessions.groupby('user_id').agg(
        session_count=('session_id', 'count'),
        avg_time_on_app=('time_on_app', 'mean'),
        avg_pages_visited=('pages_visited', 'mean'),
        session_conversion_rate=('converted', 'mean'),
    )


def build_rider_table(trips, riders, sessions):
    trip_feats, cutoff = build_trip_features(trips)
    session_feats = build_session_features(sessions)

    riders = riders.copy()
    riders['account_age_days'] = (cutoff - riders['signup_date'].dt.tz_localize('UTC')).dt.days
    riders['churn'] = (riders['churn_prob'] > 0.5).astype(int)

    table = riders.set_index('user_id').join(trip_feats).join(session_feats)

    no_trips = ['total_trips', 'total_spent', 'total_tip', 'recent_trips', 'prior_trips', 'trip_trend']
    for col in no_trips:
        table[col] = table[col].fillna(0)
    table['avg_surge'] = table['avg_surge'].fillna(1.0)
    table['recency_at_cutoff'] = table['recency_at_cutoff'].fillna(table['account_age_days'])

    no_sessions = ['session_count', 'avg_time_on_app', 'avg_pages_visited', 'session_conversion_rate']
    for col in no_sessions:
        table[col] = table[col].fillna(0)

    return table.drop(columns=['signup_date', 'churn_prob', 'referred_by']).reset_index()


def encode_features(table):
    data = table.set_index('user_id')
    data = pd.get_dummies(data, columns=data.select_dtypes(include='object').columns.tolist(), dtype=int)

    X = data.drop(columns='churn')
    y = data['churn']
    return X, y

RAW_FEATURE_COLUMNS = [
    'age', 'loyalty_status', 'city', 'avg_rating_given', 'account_age_days',
    'total_trips', 'total_spent', 'total_tip', 'avg_surge',
    'recent_trips', 'prior_trips', 'trip_trend', 'recency_at_cutoff',
    'session_count', 'avg_time_on_app', 'avg_pages_visited', 'session_conversion_rate',
]


def encode_new_rider(raw: dict, feature_columns: list) -> pd.DataFrame:
    """Encode a single rider's raw fields the same way encode_features() does,
    then align to the trained model's feature columns (missing dummy columns,
    e.g. a city that got dropped during feature selection, become 0)."""
    row = pd.DataFrame([raw], columns=RAW_FEATURE_COLUMNS)
    row = pd.get_dummies(row, columns=row.select_dtypes(include='object').columns.tolist(), dtype=int)
    return row.reindex(columns=feature_columns, fill_value=0)


def select_features(X, y, random_state=42):
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    zero_mi = mi_series[mi_series == 0].index.tolist()
    print(f"Dropped {len(zero_mi)} features with 0 mutual information: {zero_mi}")

    selected = mi_series[mi_series > 0].index.tolist()
    print(f"Features selected (MI > 0): {len(selected)}")
    return selected


if __name__ == '__main__':
    trips, riders, sessions = load_data()
    table = build_rider_table(trips, riders, sessions)
    X, y = encode_features(table)

    selected = select_features(X, y)

    model_df = X[selected].copy()
    model_df['churn'] = y.values
    model_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {model_df.shape} -> {OUTPUT_PATH}")

    with open(FEATURE_COLUMNS_PATH, 'w') as f:
        json.dump(selected, f, indent=2)
    print(f"Saved feature schema -> {FEATURE_COLUMNS_PATH}")