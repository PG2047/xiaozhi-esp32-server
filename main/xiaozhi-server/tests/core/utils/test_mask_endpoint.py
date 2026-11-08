"""接入点地址打日志前去掉 token（mask_endpoint_url / mask_endpoint_error）。"""

from core.utils.util import mask_endpoint_error, mask_endpoint_url, validate_mcp_endpoint

URL = "ws://127.0.0.1:8004/mcp_endpoint/mcp/?token=AbC%2Fde%3D%3D"


def test_drops_query_and_keeps_where_it_points():
    assert mask_endpoint_url(URL) == "ws://127.0.0.1:8004/mcp_endpoint/mcp/?…"
    assert "AbC" not in mask_endpoint_url(URL.replace("/mcp/", "/call/"))


def test_drops_userinfo_and_handles_plain_or_odd_input():
    assert mask_endpoint_url("wss://u:pw@example.com/mcp_endpoint/mcp/?token=x") == "wss://example.com/mcp_endpoint/mcp/?…"
    assert mask_endpoint_url("ws://example.com/mcp_endpoint/mcp/") == "ws://example.com/mcp_endpoint/mcp/"
    assert mask_endpoint_url(None) == "None"


def test_masked_url_is_still_valid_shape_for_logs_only():
    # 原地址照常通过校验；打日志的版本只是给人看，不拿去连接
    assert validate_mcp_endpoint(URL)


def test_error_text_hides_token_value():
    err = ValueError(f"isn't a valid URI: {URL}")
    out = mask_endpoint_error(err)
    assert "AbC" not in out and "token=***" in out
    assert mask_endpoint_error(ConnectionRefusedError("[Errno 111] Connect call failed")) == "[Errno 111] Connect call failed"
