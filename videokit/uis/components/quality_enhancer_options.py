from typing import List, Optional, Tuple

import gradio

from videokit import state_manager, translator
from videokit.common_helper import calculate_float_step, calculate_int_step
from videokit.processors.core import load_processor_module
from videokit.processors.modules.quality_enhancer import choices as quality_enhancer_choices
from videokit.processors.modules.quality_enhancer.types import QualityEnhancerModel, QualityEnhancerWeight
from videokit.uis.core import get_ui_component, register_ui_component

QUALITY_ENHANCER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
QUALITY_ENHANCER_BLEND_SLIDER : Optional[gradio.Slider] = None
QUALITY_ENHANCER_WEIGHT_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global QUALITY_ENHANCER_MODEL_DROPDOWN
	global QUALITY_ENHANCER_BLEND_SLIDER
	global QUALITY_ENHANCER_WEIGHT_SLIDER

	has_quality_enhancer = 'quality_enhancer' in state_manager.get_item('processors')
	QUALITY_ENHANCER_MODEL_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.model_dropdown', 'videokit.processors.modules.quality_enhancer'),
		choices = quality_enhancer_choices.quality_enhancer_models,
		value = state_manager.get_item('quality_enhancer_model'),
		visible = has_quality_enhancer
	)
	QUALITY_ENHANCER_BLEND_SLIDER = gradio.Slider(
		label = translator.get('uis.blend_slider', 'videokit.processors.modules.quality_enhancer'),
		value = state_manager.get_item('quality_enhancer_blend'),
		step = calculate_int_step(quality_enhancer_choices.quality_enhancer_blend_range),
		minimum = quality_enhancer_choices.quality_enhancer_blend_range[0],
		maximum = quality_enhancer_choices.quality_enhancer_blend_range[-1],
		visible = has_quality_enhancer
	)
	QUALITY_ENHANCER_WEIGHT_SLIDER = gradio.Slider(
		label = translator.get('uis.weight_slider', 'videokit.processors.modules.quality_enhancer'),
		value = state_manager.get_item('quality_enhancer_weight'),
		step = calculate_float_step(quality_enhancer_choices.quality_enhancer_weight_range),
		minimum = quality_enhancer_choices.quality_enhancer_weight_range[0],
		maximum = quality_enhancer_choices.quality_enhancer_weight_range[-1],
		visible = has_quality_enhancer and load_processor_module('quality_enhancer').get_inference_pool() and load_processor_module('quality_enhancer').has_weight_input()
	)
	register_ui_component('quality_enhancer_model_dropdown', QUALITY_ENHANCER_MODEL_DROPDOWN)
	register_ui_component('quality_enhancer_blend_slider', QUALITY_ENHANCER_BLEND_SLIDER)
	register_ui_component('quality_enhancer_weight_slider', QUALITY_ENHANCER_WEIGHT_SLIDER)


def listen() -> None:
	QUALITY_ENHANCER_MODEL_DROPDOWN.change(update_quality_enhancer_model, inputs = QUALITY_ENHANCER_MODEL_DROPDOWN, outputs = [ QUALITY_ENHANCER_MODEL_DROPDOWN, QUALITY_ENHANCER_WEIGHT_SLIDER ])
	QUALITY_ENHANCER_BLEND_SLIDER.release(update_quality_enhancer_blend, inputs = QUALITY_ENHANCER_BLEND_SLIDER)
	QUALITY_ENHANCER_WEIGHT_SLIDER.release(update_quality_enhancer_weight, inputs = QUALITY_ENHANCER_WEIGHT_SLIDER)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ QUALITY_ENHANCER_MODEL_DROPDOWN, QUALITY_ENHANCER_BLEND_SLIDER, QUALITY_ENHANCER_WEIGHT_SLIDER ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Slider, gradio.Slider]:
	has_quality_enhancer = 'quality_enhancer' in processors
	return gradio.Dropdown(visible = has_quality_enhancer), gradio.Slider(visible = has_quality_enhancer), gradio.Slider(visible = has_quality_enhancer and load_processor_module('quality_enhancer').get_inference_pool() and load_processor_module('quality_enhancer').has_weight_input())


def update_quality_enhancer_model(quality_enhancer_model : QualityEnhancerModel) -> Tuple[gradio.Dropdown, gradio.Slider]:
	quality_enhancer_module = load_processor_module('quality_enhancer')
	quality_enhancer_module.clear_inference_pool()
	state_manager.set_item('quality_enhancer_model', quality_enhancer_model)

	if quality_enhancer_module.pre_check():
		return gradio.Dropdown(value = state_manager.get_item('quality_enhancer_model')), gradio.Slider(visible = quality_enhancer_module.has_weight_input())
	return gradio.Dropdown(), gradio.Slider()


def update_quality_enhancer_blend(quality_enhancer_blend : float) -> None:
	state_manager.set_item('quality_enhancer_blend', int(quality_enhancer_blend))


def update_quality_enhancer_weight(quality_enhancer_weight : QualityEnhancerWeight) -> None:
	state_manager.set_item('quality_enhancer_weight', quality_enhancer_weight)

