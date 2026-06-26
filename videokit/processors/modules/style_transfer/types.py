from typing import Dict, List, Literal, TypeAlias, TypedDict

from videokit.types import Mask, VisionFrame

StyleTransferInputs = TypedDict('StyleTransferInputs',
{
	'reference_vision_frame' : VisionFrame,
	'source_vision_frames' : List[VisionFrame],
	'target_vision_frame' : VisionFrame,
	'temp_vision_frame' : VisionFrame,
	'temp_vision_mask' : Mask
})

StyleTransferModel = Literal['blendswap_256', 'ghost_1_256', 'ghost_2_256', 'ghost_3_256', 'hififace_unofficial_256', 'hyperswap_1a_256', 'hyperswap_1b_256', 'hyperswap_1c_256', 'inswapper_128', 'inswapper_128_fp16', 'simswap_256', 'simswap_unofficial_512', 'uniface_256']

StyleTransferWeight : TypeAlias = float

StyleTransferSet : TypeAlias = Dict[StyleTransferModel, List[str]]
