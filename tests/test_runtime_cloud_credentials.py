from __future__ import annotations

import types

from hybrid_demo import runtime


def test_get_cloud_credential_prefers_workload_identity(monkeypatch):
    runtime.unload()
    monkeypatch.setenv("AZURE_CLIENT_ID", "client-id")
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant-id")
    monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", "/tmp/fed-token")

    fake_identity = types.SimpleNamespace(
        WorkloadIdentityCredential=lambda **kwargs: ("workload", kwargs),
        ManagedIdentityCredential=lambda **kwargs: ("managed", kwargs),
        ChainedTokenCredential=lambda *creds: ("chain", creds),
    )
    monkeypatch.setitem(__import__("sys").modules, "azure.identity", fake_identity)

    credential = runtime._get_cloud_credential()
    assert credential[0] == "chain"
    assert credential[1][0][0] == "workload"
    assert credential[1][1][0] == "managed"


def test_get_cloud_chat_client_uses_cluster_credential(monkeypatch):
    runtime.unload()

    class _FakeFoundryChatClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "hybrid_demo.runtime.config.get_model",
        lambda _role: types.SimpleNamespace(
            provider="foundry",
            model="demo-model",
            endpoint=lambda: "https://example.services.ai.azure.com",
            endpoint_env=None,
        ),
    )
    monkeypatch.setattr("hybrid_demo.runtime._get_cloud_credential", lambda: "cred")
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_framework.foundry",
        types.SimpleNamespace(FoundryChatClient=_FakeFoundryChatClient),
    )

    client = runtime.get_cloud_chat_client("cloud.research")
    assert client.kwargs["credential"] == "cred"
    assert client.kwargs["model"] == "demo-model"