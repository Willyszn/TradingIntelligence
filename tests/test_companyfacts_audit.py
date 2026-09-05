from trading_intelligence.data.acquisition import sec_bulk

def test_companyfacts_fields():
    assert 'filed' in sec_bulk.COMPANYFACTS_FIELDS
    assert 'val' in sec_bulk.COMPANYFACTS_FIELDS
