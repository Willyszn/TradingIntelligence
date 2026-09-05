from trading_intelligence.data.acquisition import binance_public

def test_curl_fallback_pins_hostname_for_tls(monkeypatch):
    calls = {}
    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        class R:
            returncode = 0
            stdout = "[]"
            stderr = ""
        return R()
    monkeypatch.setattr(binance_public.subprocess, "run", fake_run)
    monkeypatch.setattr(binance_public.shutil, "which", lambda name: "curl.exe")
    payload = binance_public._curl_json(
        binance_public.BASE, host=binance_public.HOST, ip="3.114.119.137",
        params={"symbol": "BTCUSDT", "interval": "1d", "limit": 1}, timeout=10
    )
    assert payload == []
    assert "--resolve" in calls["cmd"]
    i = calls["cmd"].index("--resolve")
    assert calls["cmd"][i+1] == "data-api.binance.vision:443:3.114.119.137"
