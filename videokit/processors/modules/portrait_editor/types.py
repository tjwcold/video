from typing import Literal, TypedDict

from videokit.types import Mask, VisionFrame

PortraitEditorInputs = TypedDict('PortraitEditorInputs',
{
	'reference_vision_frame' : VisionFrame,
	'target_vision_frame' : VisionFrame,
	'temp_vision_frame' : VisionFrame,
	'temp_vision_mask' : Mask
})

PortraitEditorModel = Literal['live_portrait']
