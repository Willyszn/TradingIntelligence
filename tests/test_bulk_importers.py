from pathlib import Path
import json, zipfile
import pandas as pd

from trading_intelligence.data.acquisition.stooq_archive import import_stooq_archive
from trading_intelligence.data.acquisition.sec_bulk import import_companyfacts_zip, import_submissions_zip


def _zip_write(path: Path, name: str, content: str):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(name, content)


def test_stooq_archive_import(tmp_path):
    archive = tmp_path / 'stooq.zip'
    csv = 'Date,Open,High,Low,Close,Volume\n2020-01-02,10,11,9,10.5,100\n2020-01-03,10.5,12,10,11.5,120\n'
    _zip_write(archive, 'aapl.us.txt', csv)
    result = import_stooq_archive(archive, tmp_path / 'market')
    assert result['imported']
    df = pd.read_csv(tmp_path / 'market' / 'stooq_aapl.us_daily.csv')
    assert len(df) == 2 and df['symbol'].iloc[0] == 'aapl.us'


def test_companyfacts_import(tmp_path):
    archive = tmp_path / 'companyfacts.zip'
    payload = {
        'cik': 320193, 'entityName': 'APPLE INC',
        'facts': {'us-gaap': {'Revenue': {'label':'Revenue','description':'rev', 'units': {'USD': [
            {'start':'2024-01-01','end':'2024-12-31','filed':'2025-01-30','form':'10-K','val':100,'accn':'0000320193-25-000001'}
        ]}}}}
    }
    _zip_write(archive, 'CIK0000320193.json', json.dumps(payload))
    result = import_companyfacts_zip(archive, tmp_path / 'fund')
    assert result['rows'] == 1
    out = pd.read_csv(result['output'])
    assert out.loc[0,'tag'] == 'Revenue'


def test_submissions_import(tmp_path):
    archive = tmp_path / 'submissions.zip'
    payload = {
        'cik': 320193,
        'filings': {'recent': {
            'accessionNumber':['0000320193-25-000001'], 'filingDate':['2025-01-30'],
            'reportDate':['2024-12-31'], 'acceptanceDateTime':['2025-01-30T20:00:00.000Z'],
            'form':['10-K'], 'primaryDocument':['a10k.htm'], 'isXBRL':[1], 'items':['']
        }}
    }
    _zip_write(archive, 'CIK0000320193.json', json.dumps(payload))
    result = import_submissions_zip(archive, tmp_path / 'events')
    assert result['rows'] == 1
    out = pd.read_csv(result['output'])
    assert out.loc[0,'form'] == '10-K'


def test_streaming_importers_write_output_without_full_dataframe(tmp_path):
    # Small fixture proving incremental writer path is valid; production archives
    # use the same code path and chunking controls.
    archive = tmp_path / 'companyfacts.zip'
    payload = {
        'cik': 1, 'entityName': 'X',
        'facts': {'us-gaap': {'Revenue': {'label':'Revenue','description':'rev', 'units': {'USD': [
            {'start':'2024-01-01','end':'2024-12-31','filed':'2025-01-30','form':'10-K','val':100,'accn':'x'}
        ]}}}}
    }
    _zip_write(archive, 'CIK0000000001.json', json.dumps(payload))
    result = import_companyfacts_zip(archive, tmp_path / 'fund', chunk_rows=1)
    out = pd.read_csv(result['output'])
    assert result['rows'] == 1 and len(out) == 1


def test_stooq_d_us_txt_real_format_and_nested_path(tmp_path):
    archive = tmp_path / "d_us_txt.zip"
    text = "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\nAAPL.US,D,20240102,000000,185.64,188.44,183.89,185.64,82488700,0\nAAPL.US,D,20240103,000000,184.22,185.88,183.43,184.25,58414500,0\n"
    _zip_write(archive, "data/daily/us/nasdaq/aapl.us.txt", text)
    result = import_stooq_archive(archive, tmp_path / "market")
    assert result["imported_count"] == 1
    df = pd.read_csv(tmp_path / "market" / "stooq_aapl.us_daily.csv")
    assert len(df) == 2
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    assert df["symbol"].iloc[0] == "aapl.us"
    assert df["timestamp"].iloc[0].startswith("2024-01-02")


def test_submissions_entity_master(tmp_path):
    archive = tmp_path / 'submissions.zip'
    payload = {
        'cik': 320193, 'name': 'APPLE INC', 'tickers':['AAPL'], 'exchanges':['Nasdaq'],
        'sic':'3571', 'sicDescription':'Electronic Computers',
        'filings': {'recent': {
            'accessionNumber':['0000320193-25-000001'], 'filingDate':['2025-01-30'],
            'reportDate':['2024-12-31'], 'acceptanceDateTime':['2025-01-30T20:00:00.000Z'],
            'form':['10-K'], 'primaryDocument':['a10k.htm'], 'isXBRL':[1], 'items':['']
        }}
    }
    _zip_write(archive, 'CIK0000320193.json', json.dumps(payload))
    result = import_submissions_zip(archive, tmp_path / 'events')
    assert result['entities'] == 1
    entities = pd.read_csv(result['entity_output'])
    assert entities.loc[0, 'tickers'] == 'AAPL'
    assert int(entities.loc[0, 'cik']) == 320193


def test_companyfacts_cik_filter(tmp_path):
    archive = tmp_path / 'companyfacts.zip'
    p1 = {
        'cik': 1, 'entityName': 'X',
        'facts': {'us-gaap': {'Revenue': {'label':'Revenue','description':'rev', 'units': {'USD': [
            {'start':'2024-01-01','end':'2024-12-31','filed':'2025-01-30','form':'10-K','val':100,'accn':'x'}
        ]}}}}
    }
    p2 = {
        'cik': 2, 'entityName': 'Y',
        'facts': {'us-gaap': {'Revenue': {'label':'Revenue','description':'rev', 'units': {'USD': [
            {'start':'2024-01-01','end':'2024-12-31','filed':'2025-01-30','form':'10-K','val':200,'accn':'y'}
        ]}}}}
    }
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('CIK0000000001.json', json.dumps(p1))
        z.writestr('CIK0000000002.json', json.dumps(p2))
    result = import_companyfacts_zip(archive, tmp_path / 'fund', cik_filter={1})
    out = pd.read_csv(result['output'])
    assert result['json_files_matched'] == 1
    assert len(out) == 1 and int(out.loc[0,'cik']) == 1
