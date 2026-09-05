from pathlib import Path
import zipfile, json
import pandas as pd
from trading_intelligence.data.acquisition.sec_bulk import import_companyfacts_zip


def test_selected_companyfacts_filters_ciks(tmp_path):
    archive = tmp_path / 'companyfacts.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for cik in (1, 2):
            payload = {
                'cik': cik, 'entityName': f'E{cik}',
                'facts': {'us-gaap': {'Revenue': {'label':'Revenue','description':'rev','units': {'USD':[{'start':'2020-01-01','end':'2020-12-31','filed':'2021-01-01','form':'10-K','val':cik}]}}}}
            }
            z.writestr(f'CIK{cik:010d}.json', json.dumps(payload))
    out = tmp_path / 'out'
    res = import_companyfacts_zip(archive, out, cik_filter={2})
    assert res['json_files_matched'] == 1
    df = pd.read_csv(out/'sec_companyfacts_long.csv')
    assert len(df) == 1
    assert int(df.iloc[0].cik) == 2
