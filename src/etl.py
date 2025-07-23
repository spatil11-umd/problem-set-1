import pandas as pd
import os

def run_etl():
    os.makedirs('data', exist_ok=True)

    # These are .feather files hosted online, but you're treating them like CSVs.
    pred_universe_raw = pd.read_csv('https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1')
    arrest_events_raw = pd.read_csv('https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1')

    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw['filing_date'])
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw['filing_date'])

    pred_universe_raw.drop(columns=['filing_date'], inplace=True)
    arrest_events_raw.drop(columns=['filing_date'], inplace=True)

    pred_universe_raw.to_csv('data/pred_universe_raw.csv', index=False)
    arrest_events_raw.to_csv('data/arrest_events_raw.csv', index=False)

    print("ETL complete: Data saved to 'data/' folder.")
