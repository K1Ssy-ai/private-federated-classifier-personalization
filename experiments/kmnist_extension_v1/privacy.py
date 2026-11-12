"""Official Opacus RDP accounting. No numerical fallback, no client amplification."""
import math
from functools import lru_cache
from opacus.accountants.analysis import rdp

ORDERS = [1.25, 1.5, 1.75] + list(range(2, 65)) + [96, 128, 192, 256]

def epsilon(events, delta):
    total = sum((rdp.compute_rdp(q=q, noise_multiplier=sigma, steps=steps,
                                orders=ORDERS) for q, sigma, steps in events))
    if isinstance(total, int) and total == 0:
        return 0.0
    eps, _ = rdp.get_privacy_spent(orders=ORDERS, rdp=total, delta=delta)
    return float(eps)

@lru_cache(maxsize=128)
def calibrate(q, steps, target, delta):
    assert 0 < q <= 1 and steps > 0 and target > 0 and 0 < delta < 1
    lo, hi = 0.05, 1.0
    while epsilon([(q, hi, steps)], delta) > target:
        hi *= 2
        if hi > 1024:
            raise RuntimeError('Cannot calibrate noise')
    for _ in range(36):
        mid = (lo + hi) / 2
        if epsilon([(q, mid, steps)], delta) > target:
            lo = mid
        else:
            hi = mid
    return hi

def plan(cfg, method):
    if method == 'nonprivate':
        return {'sigma_body': 0.0, 'sigma_head': 0.0,
                'epsilon': None, 'private': False}
    steps = cfg['rounds'] * cfg['local_steps']
    if method == 'shared':
        sb = calibrate(cfg['q'], steps, cfg['epsilon'], cfg['delta'])
        return {'sigma_body': sb, 'sigma_head': 0.0,
                'epsilon': epsilon([(cfg['q'], sb, steps)], cfg['delta']),
                'private': True}
    rho = cfg['body_fraction']
    sb = calibrate(cfg['q'], steps, rho * cfg['epsilon'], cfg['delta']/2)
    sh = calibrate(cfg['q'], cfg['head_steps'], (1-rho)*cfg['epsilon'], cfg['delta']/2)
    # The separate budgets set a noise RATIO only. Joint RDP calibration uses
    # the full budget fairly, rather than wasting the slack in basic composition.
    lo,hi=0.05,1.0
    for _ in range(36):
        scale=(lo+hi)/2
        value=epsilon([(cfg['q'],sb*scale,steps),
                       (cfg['q'],sh*scale,cfg['head_steps'])],cfg['delta'])
        if value>cfg['epsilon']: lo=scale
        else: hi=scale
    sb*=hi; sh*=hi
    return {'sigma_body': sb, 'sigma_head': sh,
            'epsilon': epsilon([(cfg['q'], sb, steps),
                                (cfg['q'], sh, cfg['head_steps'])], cfg['delta']),
            'noise_ratio_reference_fraction':rho,'joint_scale':hi,'private':True}
