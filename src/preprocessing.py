import pandas as pd
from datetime import timedelta
import os

def preprocess():
    os.makedirs('data', exist_ok=True)
    print("hi")

    # Load CSVs and parse date columns
    pred_universe = pd.read_csv('data/pred_universe_raw.csv', parse_dates=['arrest_date_univ'])
    arrest_events = pd.read_csv('data/arrest_events_raw.csv', parse_dates=['arrest_date_event'])

    # Rename for clarity before merge
    pred_universe = pred_universe.rename(columns={'arrest_date_univ': 'arrest_date_current'})
    arrest_events = arrest_events.rename(columns={
        'arrest_date_event': 'arrest_date_history',
        'charge_degree': 'charge_class_history'
    })

    # Merge on person_id to align current + historical arrests
    df_arrests = pd.merge(pred_universe, arrest_events, on='person_id', how='outer')

    # Label: was the person rearrested for a felony within 1 year?
    print(f"Number of rows with missing current arrest date: {df_arrests['arrest_date_current'].isna().sum()}")
    df_arrests = df_arrests.dropna(subset=['arrest_date_current'])

    print("Unique charge_class_history values:", df_arrests['charge_class_history'].unique())

    def was_rearrested(group):
        current_date = group['arrest_date_current'].iloc[0]
        if pd.isna(current_date):
            return 0
        mask = (
            (group['arrest_date_history'] > current_date) &
            (group['arrest_date_history'] <= current_date + timedelta(days=365)) &
            (group['charge_class_history'].str.lower() == 'felony')
        )
        return int(mask.any())

    y_series = df_arrests.groupby('person_id').apply(was_rearrested).rename('y').reset_index()
    df_arrests = pd.merge(df_arrests, y_series, on='person_id', how='left')

    print("What share of arrestees were rearrested for a felony in the next year?")
    share_rearrested = df_arrests[['person_id', 'y']].drop_duplicates()['y'].mean()
    print(f"{share_rearrested:.2%}")

    print("df_arrests columns:", df_arrests.columns)

    # Determine current charge class using the original pred_universe arrest_id
    df_arrests = pd.merge(
        df_arrests,
        arrest_events[['arrest_id', 'charge_class_history']].rename(columns={
            'arrest_id': 'arrest_id_x',
            'charge_class_history': 'charge_class_current'
        }),
        on='arrest_id_x',
        how='left'
    )

    df_arrests['current_charge_felony'] = (df_arrests['charge_class_current'].str.lower() == 'felony').astype(int)

    print("What share of current charges are felonies?")
    print(f"{df_arrests['current_charge_felony'].mean():.2%}")

    # Count felony arrests in past year before current arrest
    def felony_arrests_last_year(group):
        current_date = group['arrest_date_current'].iloc[0]
        if pd.isna(current_date):
            return 0
        mask = (
            (group['arrest_date_history'] < current_date) &
            (group['arrest_date_history'] >= current_date - timedelta(days=365)) &
            (group['charge_class_history'].str.lower() == 'felony')
        )
        return mask.sum()

    fel_counts = df_arrests.groupby('person_id').apply(felony_arrests_last_year).rename('num_fel_arrests_last_year').reset_index()
    df_arrests = pd.merge(df_arrests, fel_counts, on='person_id', how='left')

    print("Average number of felony arrests in the last year:")
    print(df_arrests['num_fel_arrests_last_year'].mean())

    # Add features back to pred_universe for modeling
    pred_universe = pd.merge(
        pred_universe,
        df_arrests[['person_id', 'y', 'current_charge_felony', 'num_fel_arrests_last_year']].drop_duplicates('person_id'),
        on='person_id',
        how='left'
    )

    print("Mean of 'num_fel_arrests_last_year':")
    print(pred_universe['num_fel_arrests_last_year'].mean())
    print(pred_universe.head())

    # Save final dataframe
    df_arrests.to_csv('data/preprocessed_arrests.csv', index=False)

    return df_arrests
