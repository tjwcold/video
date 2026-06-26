from typing import Literal, TypedDict

from videokit.types import Mask, VisionFrame

RegionDebuggerInputs = TypedDict('RegionDebuggerInputs',
{
	'reference_vision_frame' : VisionFrame,
	'target_vision_frame' : VisionFrame,
	'temp_vision_frame' : VisionFrame,
	'temp_vision_mask' : Mask
})

RegionDebuggerItem = Literal['bounding-box', 'region-landmark-5', 'region-landmark-5/68', 'region-landmark-68', 'region-landmark-68/5', 'region-mask']
