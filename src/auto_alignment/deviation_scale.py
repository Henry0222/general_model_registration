from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
REFERENCE_15_RGB = np.asarray([(0, 1, 130), (0, 43, 255), (0, 85, 255), (0, 128, 255), (0, 171, 255), (0, 213, 255), (0, 255, 251), (2, 255, 1), (255, 255, 0), (255, 212, 0), (255, 170, 0), (255, 127, 0), (255, 84, 0), (254, 42, 0), (204, 0, 0)], dtype=np.uint8)

@dataclass(frozen=True)
class DeviationScale:
    """A result-level asymmetric 15-segment signed-deviation spectrum."""
    minimum_critical_mm: float
    minimum_nominal_mm: float
    maximum_nominal_mm: float
    maximum_critical_mm: float
    configured_minimum_nominal_mm: float
    configured_maximum_nominal_mm: float
    colors_rgb: np.ndarray = field(default_factory=lambda: REFERENCE_15_RGB.copy(), repr=False)

    def __post_init__(self) -> None:
        values = np.asarray([self.minimum_critical_mm, self.minimum_nominal_mm, self.maximum_nominal_mm, self.maximum_critical_mm, self.configured_minimum_nominal_mm, self.configured_maximum_nominal_mm], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('偏差色阶边界必须是有限毫米值。')
        if self.configured_minimum_nominal_mm > 0:
            raise ValueError('最小名义值必须小于或等于 0。')
        if self.configured_maximum_nominal_mm < 0:
            raise ValueError('最大名义值必须大于或等于 0。')
        if not self.minimum_critical_mm <= self.minimum_nominal_mm <= self.maximum_nominal_mm <= self.maximum_critical_mm:
            raise ValueError('偏差色阶边界顺序无效。')
        colors = np.asarray(self.colors_rgb)
        if colors.shape != (15, 3):
            raise ValueError('偏差色阶必须正好包含 15 个 RGB 色段。')
        if np.any(colors < 0) or np.any(colors > 255):
            raise ValueError('偏差色阶 RGB 必须位于 0–255。')
        object.__setattr__(self, 'colors_rgb', colors.astype(np.uint8, copy=True))

    @classmethod
    def from_signed_distances(cls, signed_distances_mm: np.ndarray, minimum_nominal_mm: float=-0.05, maximum_nominal_mm: float=0.05, *, fallback_limit_mm: float | None=None) -> DeviationScale:
        values = np.asarray(signed_distances_mm, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        minimum_nominal = float(minimum_nominal_mm)
        maximum_nominal = float(maximum_nominal_mm)
        if minimum_nominal > 0 or maximum_nominal < 0:
            raise ValueError('名义范围必须包含 0 mm。')
        if minimum_nominal >= maximum_nominal:
            raise ValueError('最小名义值必须小于最大名义值。')
        fallback = float(fallback_limit_mm) if fallback_limit_mm is not None else max(abs(minimum_nominal), abs(maximum_nominal), 0.15)
        if not np.isfinite(fallback) or fallback <= 0:
            raise ValueError('备用色阶范围必须是正有限毫米值。')
        if len(values):
            minimum_critical = min(float(np.min(values)), 0.0)
            maximum_critical = max(float(np.max(values)), 0.0)
        else:
            minimum_critical = -fallback
            maximum_critical = fallback
        effective_minimum_nominal = float(np.clip(minimum_nominal, minimum_critical, 0.0))
        effective_maximum_nominal = float(np.clip(maximum_nominal, 0.0, maximum_critical))
        return cls(minimum_critical_mm=minimum_critical, minimum_nominal_mm=effective_minimum_nominal, maximum_nominal_mm=effective_maximum_nominal, maximum_critical_mm=maximum_critical, configured_minimum_nominal_mm=minimum_nominal, configured_maximum_nominal_mm=maximum_nominal)

    @property
    def colors_float(self) -> np.ndarray:
        return self.colors_rgb.astype(float) / 255.0

    def with_critical_limits(self, minimum_critical_mm: float, maximum_critical_mm: float) -> DeviationScale:
        minimum = float(minimum_critical_mm)
        maximum = float(maximum_critical_mm)
        if not np.isfinite([minimum, maximum]).all():
            raise ValueError('偏差临界值必须是有限毫米值。')
        if minimum > 0 or maximum < 0:
            raise ValueError('最小临界值必须不大于 0，最大临界值必须不小于 0。')
        return DeviationScale(minimum_critical_mm=minimum, minimum_nominal_mm=float(np.clip(self.configured_minimum_nominal_mm, minimum, 0.0)), maximum_nominal_mm=float(np.clip(self.configured_maximum_nominal_mm, 0.0, maximum)), maximum_critical_mm=maximum, configured_minimum_nominal_mm=self.configured_minimum_nominal_mm, configured_maximum_nominal_mm=self.configured_maximum_nominal_mm, colors_rgb=self.colors_rgb)

    @property
    def segment_bounds_mm(self) -> tuple[tuple[float, float], ...]:
        negative = np.linspace(self.minimum_critical_mm, self.minimum_nominal_mm, 8)
        positive = np.linspace(self.maximum_nominal_mm, self.maximum_critical_mm, 8)
        bounds = [(float(negative[index]), float(negative[index + 1])) for index in range(7)]
        bounds.append((self.minimum_nominal_mm, self.maximum_nominal_mm))
        bounds.extend(((float(positive[index]), float(positive[index + 1])) for index in range(7)))
        return tuple(bounds)

    def map_colors(self, signed_distances_mm: np.ndarray) -> np.ndarray:
        """Map signed deviations with continuous interpolation between anchors."""
        values = np.asarray(signed_distances_mm, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('色阶映射值必须是有限数值。')
        palette = self.colors_float
        result = np.empty(values.shape + (3,), dtype=float)
        result[...] = palette[7]
        negative = values < self.minimum_nominal_mm
        negative_span = self.minimum_nominal_mm - self.minimum_critical_mm
        if np.any(negative) and negative_span > 1e-12:
            position = np.clip(
                (values[negative] - self.minimum_critical_mm) / negative_span,
                0.0,
                1.0,
            ) * 7.0
            lower = np.floor(position).astype(np.int64)
            upper = np.minimum(lower + 1, 7)
            mix = (position - lower)[..., None]
            result[negative] = palette[lower] * (1.0 - mix) + palette[upper] * mix
        elif np.any(negative):
            result[negative] = palette[0]
        positive = values > self.maximum_nominal_mm
        positive_span = self.maximum_critical_mm - self.maximum_nominal_mm
        if np.any(positive) and positive_span > 1e-12:
            position = 7.0 + np.clip(
                (values[positive] - self.maximum_nominal_mm) / positive_span,
                0.0,
                1.0,
            ) * 7.0
            lower = np.floor(position).astype(np.int64)
            upper = np.minimum(lower + 1, 14)
            mix = (position - lower)[..., None]
            result[positive] = palette[lower] * (1.0 - mix) + palette[upper] * mix
        elif np.any(positive):
            result[positive] = palette[14]
        return result

    @property
    def legend_ticks_mm(self) -> tuple[float, ...]:
        """Geomagic-style labels: six intervals per side and a green band."""
        positive = np.linspace(
            self.maximum_critical_mm,
            self.maximum_nominal_mm,
            7,
        )
        negative = np.linspace(
            self.minimum_nominal_mm,
            self.minimum_critical_mm,
            7,
        )
        return tuple(float(value) for value in np.concatenate((positive, negative)))

    def as_dict(self) -> dict[str, object]:
        return {'segments': 15, 'mapping': 'continuous_piecewise_linear', 'minimum_critical_mm': self.minimum_critical_mm, 'minimum_nominal_mm': self.minimum_nominal_mm, 'maximum_nominal_mm': self.maximum_nominal_mm, 'maximum_critical_mm': self.maximum_critical_mm, 'configured_minimum_nominal_mm': self.configured_minimum_nominal_mm, 'configured_maximum_nominal_mm': self.configured_maximum_nominal_mm, 'colors_rgb_negative_to_positive': self.colors_rgb.tolist(), 'segment_bounds_mm_negative_to_positive': [list(values) for values in self.segment_bounds_mm], 'legend_ticks_mm_positive_to_negative': list(self.legend_ticks_mm), 'positive_meaning': 'under-preparation', 'negative_meaning': 'over-preparation'}
