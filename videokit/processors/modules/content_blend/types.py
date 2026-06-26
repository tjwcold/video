from typing import Any, TypeAlias, TypedDict

from numpy.typing import NDArray

from videokit.types import Mask, VisionFrame

ContentBlendInputs = TypedDict('ContentBlendInputs',
{
	'reference_vision_frame' : VisionFrame,
	'target_vision_frame' : VisionFrame,
	'temp_vision_frame' : VisionFrame,
	'temp_vision_mask' : Mask
})

ContentBlendModel : TypeAlias = str

ContentBlendMorph : TypeAlias = NDArray[Any]
