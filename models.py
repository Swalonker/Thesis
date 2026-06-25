"""
Volatility model zoo: four models with a uniform API.

    fit(y, n_restarts)              -> dict(params, loglik, AIC, BIC, converged)
    filter(params, y)              -> sigma2 array (length T)
    forecast_next(params, history) -> (sigma2_next, density_kwargs)

Conventions
-----------
y_t is in percentage points (r_t = 100*(log P_t - log P_{t-1})), zero conditional mean.
For GARCH-G and GARCH-t, sigma2 is the conditional VARIANCE.
For t-GAS and Beta-t-EGARCH, sigma2 is the SCALE of the Student-t (variance = scale*nu/(nu-2)).

The log-variance recursion of the score-driven models is clipped to [-20, 20]
to prevent exp() overflow on pathological parameter draws.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.special import gammaln

EPS = 1e-8
THETA_CLIP = 20.0          # |log sigma^2| ceiling -> sigma^2 in [2e-9, 4.9e8]


def _gauss_loglik_obs(y, sigma2):
    sigma2 = np.maximum(sigma2, EPS)
    return -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + y ** 2 / sigma2)


def _t_loglik_obs(y, sigma2, nu):
    """Log-density of zero-mean Student-t with scale sigma^2 and nu df."""
    sigma2 = np.maximum(sigma2, EPS)
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
    return c - 0.5 * np.log(sigma2) - ((nu + 1) / 2) * np.log1p(y ** 2 / (nu * sigma2))


def _ic(loglik, k, T):
    return {"AIC": -2 * loglik + 2 * k, "BIC": -2 * loglik + np.log(T) * k}


@dataclass
class GarchGauss:
    name = "GARCH-G"; param_names = ("omega", "alpha", "beta")

    @staticmethod
    def filter(params, y):
        omega, alpha, beta = params; T = len(y); s2 = np.empty(T); s2[0] = max(np.var(y), EPS)
        for t in range(T - 1):
            s2[t + 1] = omega + alpha * y[t] ** 2 + beta * s2[t]
            if s2[t + 1] <= EPS: s2[t + 1] = EPS
        return s2

    @classmethod
    def _negloglik(cls, params, y):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999: return 1e10
        ll = _gauss_loglik_obs(y, cls.filter(params, y)).sum()
        return -ll if np.isfinite(ll) else 1e10

    @classmethod
    def fit(cls, y, n_restarts=5, seed=1):
        T = len(y); bounds = [(1e-6, 5.0), (1e-6, 0.5), (1e-6, 0.9999)]; rng = np.random.default_rng(seed)
        starts = [np.array([0.05, 0.05, 0.90])] + [np.array([rng.uniform(0.01, 0.2), rng.uniform(0.02, 0.2), rng.uniform(0.6, 0.95)]) for _ in range(n_restarts - 1)]
        best = None
        for x0 in starts:
            r = minimize(cls._negloglik, x0, args=(y,), method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000, "ftol": 1e-12})
            if best is None or r.fun < best.fun: best = r
        ll = -best.fun
        return {"model": cls.name, "params": dict(zip(cls.param_names, best.x)), "loglik": ll, **_ic(ll, 3, T), "converged": best.success}

    @classmethod
    def forecast_next(cls, params, history):
        s2 = cls.filter(params, history); omega, alpha, beta = params
        return omega + alpha * history[-1] ** 2 + beta * s2[-1], {}


@dataclass
class GarchT:
    name = "GARCH-t"; param_names = ("omega", "alpha", "beta", "nu")

    @staticmethod
    def filter(params, y):
        omega, alpha, beta, _ = params; T = len(y); s2 = np.empty(T); s2[0] = max(np.var(y), EPS)
        for t in range(T - 1):
            s2[t + 1] = omega + alpha * y[t] ** 2 + beta * s2[t]
            if s2[t + 1] <= EPS: s2[t + 1] = EPS
        return s2

    @classmethod
    def _negloglik(cls, params, y):
        omega, alpha, beta, nu = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999 or nu <= 2.05 or nu > 200: return 1e10
        s2 = cls.filter(params, y); scale2 = (nu - 2) / nu * s2          # unit-variance standardised t
        ll = _t_loglik_obs(y, scale2, nu).sum()
        return -ll if np.isfinite(ll) else 1e10

    @classmethod
    def fit(cls, y, n_restarts=5, seed=1):
        T = len(y); bounds = [(1e-6, 5.0), (1e-6, 0.5), (1e-6, 0.9999), (2.05, 100)]; rng = np.random.default_rng(seed)
        starts = [np.array([0.05, 0.05, 0.90, 8.0])] + [np.array([rng.uniform(0.01, 0.2), rng.uniform(0.02, 0.2), rng.uniform(0.6, 0.95), rng.uniform(3, 30)]) for _ in range(n_restarts - 1)]
        best = None
        for x0 in starts:
            r = minimize(cls._negloglik, x0, args=(y,), method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000, "ftol": 1e-12})
            if best is None or r.fun < best.fun: best = r
        ll = -best.fun
        return {"model": cls.name, "params": dict(zip(cls.param_names, best.x)), "loglik": ll, **_ic(ll, 4, T), "converged": best.success}

    @classmethod
    def forecast_next(cls, params, history):
        s2 = cls.filter(params, history); omega, alpha, beta, nu = params
        return omega + alpha * history[-1] ** 2 + beta * s2[-1], {"nu": nu, "dist": "t"}


@dataclass
class StudenttGas:
    name = "t-GAS"; param_names = ("omega", "A", "B", "nu")

    @staticmethod
    def _score(y, s2, nu):
        s2 = max(s2, EPS); u = y * y / s2
        return (nu + 3) / (nu + 1) * ((nu + 1) * u / (nu + u) - 1.0)

    @classmethod
    def filter(cls, params, y):
        omega, A, B, nu = params; T = len(y); th = np.empty(T); s2 = np.empty(T)
        th[0] = np.clip(omega / (1 - B) if abs(B) < 1 else np.log(max(np.var(y), EPS)), -THETA_CLIP, THETA_CLIP); s2[0] = np.exp(th[0])
        for t in range(T - 1):
            s = cls._score(y[t], s2[t], nu)
            th[t + 1] = np.clip(omega + A * s + B * th[t], -THETA_CLIP, THETA_CLIP); s2[t + 1] = np.exp(th[t + 1])
        return s2

    @classmethod
    def _negloglik(cls, params, y):
        omega, A, B, nu = params
        if abs(B) >= 1 or nu <= 2.05 or nu > 200 or A < 0: return 1e10
        ll = _t_loglik_obs(y, cls.filter(params, y), nu).sum()
        return -ll if np.isfinite(ll) else 1e10

    @classmethod
    def fit(cls, y, n_restarts=5, seed=1):
        T = len(y); bounds = [(-5, 5), (1e-6, 1.0), (-0.999, 0.999), (2.05, 100)]; rng = np.random.default_rng(seed)
        starts = [np.array([0.02, 0.05, 0.95, 8.0])] + [np.array([rng.uniform(-0.2, 0.2), rng.uniform(0.01, 0.2), rng.uniform(0.7, 0.99), rng.uniform(3, 30)]) for _ in range(n_restarts - 1)]
        best = None
        for x0 in starts:
            r = minimize(cls._negloglik, x0, args=(y,), method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000, "ftol": 1e-12})
            if best is None or r.fun < best.fun: best = r
        ll = -best.fun
        return {"model": cls.name, "params": dict(zip(cls.param_names, best.x)), "loglik": ll, **_ic(ll, 4, T), "converged": best.success}

    @classmethod
    def forecast_next(cls, params, history):
        s2 = cls.filter(params, history); omega, A, B, nu = params
        th_last = np.log(s2[-1]); s_last = cls._score(history[-1], s2[-1], nu)
        return float(np.exp(np.clip(omega + A * s_last + B * th_last, -THETA_CLIP, THETA_CLIP))), {"nu": nu, "dist": "t_scale"}


@dataclass
class BetaTEGarch:
    name = "Beta-t-EGARCH"; param_names = ("omega", "A", "A_minus", "B", "nu")

    @staticmethod
    def _score(y, s2, nu):
        s2 = max(s2, EPS); u = y * y / s2
        return (nu + 3) / (nu + 1) * ((nu + 1) * u / (nu + u) - 1.0)

    @classmethod
    def filter(cls, params, y):
        omega, A, Am, B, nu = params; T = len(y); th = np.empty(T); s2 = np.empty(T)
        th[0] = np.clip(omega / (1 - B) if abs(B) < 1 else np.log(max(np.var(y), EPS)), -THETA_CLIP, THETA_CLIP); s2[0] = np.exp(th[0])
        for t in range(T - 1):
            s = cls._score(y[t], s2[t], nu); asym = (A + Am * (y[t] < 0)) * s
            th[t + 1] = np.clip(omega + asym + B * th[t], -THETA_CLIP, THETA_CLIP); s2[t + 1] = np.exp(th[t + 1])
        return s2

    @classmethod
    def _negloglik(cls, params, y):
        omega, A, Am, B, nu = params
        if abs(B) >= 1 or nu <= 2.05 or nu > 200 or A < 0: return 1e10
        ll = _t_loglik_obs(y, cls.filter(params, y), nu).sum()
        return -ll if np.isfinite(ll) else 1e10

    @classmethod
    def fit(cls, y, n_restarts=5, seed=1):
        T = len(y); bounds = [(-5, 5), (1e-6, 1.0), (-0.5, 0.5), (-0.999, 0.999), (2.05, 100)]; rng = np.random.default_rng(seed)
        starts = [np.array([0.02, 0.04, 0.02, 0.95, 8.0])] + [np.array([rng.uniform(-0.2, 0.2), rng.uniform(0.01, 0.2), rng.uniform(-0.1, 0.1), rng.uniform(0.7, 0.99), rng.uniform(3, 30)]) for _ in range(n_restarts - 1)]
        best = None
        for x0 in starts:
            r = minimize(cls._negloglik, x0, args=(y,), method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000, "ftol": 1e-12})
            if best is None or r.fun < best.fun: best = r
        ll = -best.fun
        return {"model": cls.name, "params": dict(zip(cls.param_names, best.x)), "loglik": ll, **_ic(ll, 5, T), "converged": best.success}

    @classmethod
    def forecast_next(cls, params, history):
        s2 = cls.filter(params, history); omega, A, Am, B, nu = params
        th_last = np.log(s2[-1]); s = cls._score(history[-1], s2[-1], nu); asym = (A + Am * (history[-1] < 0)) * s
        return float(np.exp(np.clip(omega + asym + B * th_last, -THETA_CLIP, THETA_CLIP))), {"nu": nu, "dist": "t_scale"}


MODELS = (GarchGauss, GarchT, StudenttGas, BetaTEGarch)
