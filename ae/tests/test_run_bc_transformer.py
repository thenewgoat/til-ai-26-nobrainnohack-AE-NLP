"""run_bc_transformer._config_from_env reads model scale from env vars."""
from run_bc_transformer import _config_from_env


def test_config_defaults(monkeypatch):
    for v in ("TF_D_MODEL", "TF_N_LAYERS", "TF_N_HEADS", "TF_FFN_DIM",
              "TF_DROPOUT"):
        monkeypatch.delenv(v, raising=False)
    assert _config_from_env() == {"d_model": 64, "n_layers": 4,
                                  "n_heads": 4, "dropout": 0.1}


def test_config_from_env_overrides(monkeypatch):
    monkeypatch.setenv("TF_D_MODEL", "128")
    monkeypatch.setenv("TF_N_LAYERS", "6")
    monkeypatch.setenv("TF_FFN_DIM", "384")
    cfg = _config_from_env()
    assert cfg["d_model"] == 128
    assert cfg["n_layers"] == 6
    assert cfg["ffn_dim"] == 384
