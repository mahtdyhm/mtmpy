"""
KinetiQ OS - Biomechanics One-Euro Kinematic Filter
فیلتر تطبیقی وان-یورو برای حذف لرزش (Jitter) مفاصل بدون ایجاد تاخیر (Lag)
"""

import math
from typing import Optional


class LowPassFilter:
    """فیلتر پایین‌گذر مرتبه اول"""
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.s: Optional[float] = None

    def filter(self, val: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self.s is None:
            self.s = val
        else:
            self.s = self.alpha * val + (1.0 - self.alpha) * self.s
        return self.s

    def reset(self):
        self.s = None


class OneEuroFilter:
    """
    فیلتر تطبیقی 1-Euro بر اساس فرکانس و مشتق سرعت:
    در سرعت‌های پایین، نویز را به شدت فیلتر می‌کند تا مفاصل نلرزند؛
    در حرکات سریع (پرش، شوت)، تاخیر را به صفر می‌رساند تا حرکت دقیق بماند.
    """
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate if rate > 0 else 0.001
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: float) -> float:
        if self.last_time is None:
            self.last_time = timestamp
            return self.x_filt.filter(x, 1.0)

        dt = timestamp - self.last_time
        self.last_time = timestamp
        rate = 1.0 / dt if dt > 0.0 else 30.0

        # محاسبه مشتق سرعت تغییر نقطه
        dx = (x - (self.x_filt.s if self.x_filt.s is not None else x)) * rate
        edx = self.dx_filt.filter(dx, self._alpha(rate, self.d_cutoff))

        # فرکانس قطع تطبیقی بر اساس سرعت
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filt.filter(x, self._alpha(rate, cutoff))

    def reset(self):
        self.x_filt.reset()
        self.dx_filt.reset()
        self.last_time = None
