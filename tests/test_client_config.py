import pytest
from pydantic import ValidationError

from verifiers.types import ClientConfig, EndpointClientConfig


def test_client_config_allows_leaf_endpoint_configs():
    config = ClientConfig(
        api_base_url="http://localhost:8000/v1",
        endpoint_configs=[
            EndpointClientConfig(api_base_url="http://localhost:8001/v1"),
            {"api_base_url": "http://localhost:8002/v1"},
        ],
    )

    assert len(config.endpoint_configs) == 2
    assert config.endpoint_configs[0].api_base_url == "http://localhost:8001/v1"
    assert config.endpoint_configs[1].api_base_url == "http://localhost:8002/v1"


def test_client_config_rejects_recursive_endpoint_configs():
    with pytest.raises(ValidationError, match="cannot include endpoint_configs"):
        ClientConfig.model_validate(
            {
                "api_base_url": "http://localhost:8000/v1",
                "endpoint_configs": [
                    {
                        "api_base_url": "http://localhost:8001/v1",
                        "endpoint_configs": [
                            {"api_base_url": "http://localhost:8002/v1"}
                        ],
                    }
                ],
            }
        )


def test_client_config_accepts_empty_nested_endpoint_configs_key():
    config = ClientConfig.model_validate(
        {
            "api_base_url": "http://localhost:8000/v1",
            "endpoint_configs": [
                {
                    "api_base_url": "http://localhost:8001/v1",
                    "endpoint_configs": [],
                }
            ],
        }
    )

    assert len(config.endpoint_configs) == 1
    assert config.endpoint_configs[0].api_base_url == "http://localhost:8001/v1"


def test_build_http_client_binds_ipv4_and_keeps_limits():
    """_build_http_client must bind the source address to IPv4 AND keep the
    configured pool limits.

    On dual-stack compute nodes with dead IPv6 egress, httpx tries the AAAA
    record first and hangs a connect attempt per call before IPv4 fallback;
    across many workers that exhausts connections and surfaces as
    grader_error=1.0. We pass an explicit AsyncHTTPTransport(local_address=
    "0.0.0.0"). Because an explicit transport makes httpx IGNORE the client-level
    ``limits=``, the limits must be set on the transport -- this test guards both.
    """
    from verifiers.types import ClientConfig
    from verifiers.utils.client_utils import _build_http_client

    cfg = ClientConfig(
        api_base_url="http://example.invalid",
        max_connections=7,
        max_keepalive_connections=3,
    )
    client = _build_http_client(cfg, {})
    pool = client._transport._pool

    assert pool._local_address == "0.0.0.0", "source address must be bound to IPv4"
    assert pool._max_connections == 7, (
        "configured pool limits must survive the explicit transport "
        "(httpx drops client-level limits= when transport= is passed)"
    )
