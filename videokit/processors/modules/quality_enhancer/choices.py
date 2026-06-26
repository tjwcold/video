from typing import List, Sequence, get_args

from videokit.common_helper import create_float_range, create_int_range
from videokit.processors.modules.quality_enhancer.types import QualityEnhancerModel

quality_enhancer_models : List[QualityEnhancerModel] = list(get_args(QualityEnhancerModel))

quality_enhancer_blend_range : Sequence[int] = create_int_range(0, 100, 1)

quality_enhancer_weight_range : Sequence[float] = create_float_range(0.0, 1.0, 0.05)
