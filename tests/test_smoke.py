import torch

from zemlya import PRESETS, Zemlya, ZemlyaConfig
from zemlya.data import SyntheticControlDataset


def _cfg():
    return ZemlyaConfig.preset("nano", context_len=128, max_horizon_patches=4)


def _batch(cfg, B=4):
    ds = SyntheticControlDataset(B, cfg.context_len, 4, cfg.patch_len, cfg.channels, cfg.action_dim)
    return {k: torch.stack([ds[i][k] for i in range(B)]) for k in ds[0]}


def test_budget_under_500m():
    for name in PRESETS:
        with torch.device("meta"):
            for ch in (1, 8):
                assert Zemlya(ZemlyaConfig.preset(name, channels=ch)).param_counts()["deployable"] < 500e6, (name, ch)


def test_train_step_and_grads():
    cfg = _cfg()
    m = Zemlya(cfg)
    loss, mets = m(_batch(cfg))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in m.target_encoder.parameters())   # EMA gets no grads
    assert all(p.grad is not None for p in m.predictor.parameters())
    m.update_target(0.99)


def test_forecast_shape():
    cfg = _cfg()
    m = Zemlya(cfg).eval()
    q = m.forecast(torch.randn(2, cfg.context_len, cfg.channels), horizon_steps=70)
    assert q.shape == (2, 70, cfg.channels, len(cfg.quantiles))


def test_decide_in_bounds():
    cfg = _cfg()
    m = Zemlya(cfg).eval()
    out = m.decide(torch.randn(3, cfg.context_len, cfg.channels), horizon_patches=3, num_samples=16, iters=2)
    assert out["action"].shape == (3, cfg.action_dim) and out["plan"].shape == (3, 3, cfg.action_dim)
    assert out["plan"].min() >= cfg.action_low and out["plan"].max() <= cfg.action_high


def test_strip_for_inference():
    m = Zemlya(_cfg()).strip_for_inference().eval()
    assert m.target_encoder is None
    m.decide(torch.randn(1, 128, 1), horizon_patches=2, num_samples=8, iters=1)


def test_multivariate_variate_attention():
    cfg = ZemlyaConfig.preset("nano", context_len=128, max_horizon_patches=4, channels=3, action_dim=2)
    m = Zemlya(cfg)
    assert len(m.encoder.variate) > 0
    loss, _ = m(_batch(cfg))
    loss.backward()
    assert all(p.grad is not None for p in m.encoder.variate.parameters())
    q = m.eval().forecast(torch.randn(2, 128, 3), 40)
    assert q.shape == (2, 40, 3, 9)


def test_quantiles_monotone():
    cfg = _cfg()
    q = Zemlya(cfg).eval().forecast(torch.randn(2, 128, 1), 32)
    assert (q[..., 1:] >= q[..., :-1]).all()


def test_sigreg_regularizer_trains():
    cfg = ZemlyaConfig.preset("nano", context_len=128, max_horizon_patches=4, regularizer="sigreg")
    loss, mets = Zemlya(cfg)(_batch(cfg))
    loss.backward()
    assert "sigreg" in mets and torch.isfinite(loss)


def test_goal_conditioned_plan_and_warm_start():
    cfg = _cfg()
    m = Zemlya(cfg).eval()
    ctx = torch.randn(2, 128, 1)
    goal = torch.zeros(2, 3 * cfg.patch_len, 1)
    out = m.decide(ctx, horizon_patches=3, num_samples=16, iters=2, goal_future=goal, goal_weight=2.0)
    out2 = m.decide(ctx, horizon_patches=3, num_samples=16, iters=1, init_plan=out["plan"])
    assert out["plan"].shape == out2["plan"].shape == (2, 3, 1)


def test_critic_value_and_return_conditioning_train():
    cfg = _cfg()
    m = Zemlya(cfg)
    loss, mets = m(_batch(cfg))
    assert "value" in mets and "reward" in mets
    loss.backward()
    assert all(p.grad is not None for p in m.critic.parameters())
    assert m.policy_head.ret_embed.weight.grad is not None


def test_modes_policy_plan_auto():
    cfg = _cfg()
    m = Zemlya(cfg).eval()
    ctx = torch.randn(4, 128, 1)
    p = m.decide(ctx, mode="policy", horizon_patches=3, target_return=-1.0)
    assert p["mode"] == "policy" and p["uncertainty"].shape == (4,)
    a = m.decide(ctx, mode="auto", uncertainty_threshold=-1.0, horizon_patches=3, num_samples=8, iters=1)
    assert a["searched"].all() and a["plan"].shape == (4, 3, 1)
    n = m.decide(ctx, mode="auto", uncertainty_threshold=1e9, horizon_patches=3)
    assert not n["searched"].any()


def test_symlog_roundtrip():
    from zemlya.losses import symexp, symlog
    x = torch.tensor([-1000.0, -1.0, 0.0, 0.3, 50.0])
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-4, atol=1e-4)
