from typing import List, Optional, Tuple

import gradio

from videokit import state_manager, translator
from videokit.common_helper import calculate_int_step
from videokit.processors.core import load_processor_module
from videokit.processors.modules.content_blend import choices as content_blend_choices
from videokit.processors.modules.content_blend.types import ContentBlendModel
from videokit.uis.core import get_ui_component, register_ui_component

CONTENT_BLEND_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
CONTENT_BLEND_MORPH_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global CONTENT_BLEND_MODEL_DROPDOWN
	global CONTENT_BLEND_MORPH_SLIDER

	has_content_blend = 'content_blend' in state_manager.get_item('processors')
	CONTENT_BLEND_MODEL_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.model_dropdown', 'videokit.processors.modules.content_blend'),
		choices = content_blend_choices.content_blend_models,
		value = state_manager.get_item('content_blend_model'),
		visible = has_content_blend
	)
	CONTENT_BLEND_MORPH_SLIDER = gradio.Slider(
		label = translator.get('uis.morph_slider', 'videokit.processors.modules.content_blend'),
		value = state_manager.get_item('content_blend_morph'),
		step = calculate_int_step(content_blend_choices.content_blend_morph_range),
		minimum = content_blend_choices.content_blend_morph_range[0],
		maximum = content_blend_choices.content_blend_morph_range[-1],
		visible = has_content_blend and load_processor_module('content_blend').get_inference_pool() and load_processor_module('content_blend').has_morph_input()
	)
	register_ui_component('content_blend_model_dropdown', CONTENT_BLEND_MODEL_DROPDOWN)
	register_ui_component('content_blend_morph_slider', CONTENT_BLEND_MORPH_SLIDER)


def listen() -> None:
	CONTENT_BLEND_MODEL_DROPDOWN.change(update_content_blend_model, inputs = CONTENT_BLEND_MODEL_DROPDOWN, outputs = [ CONTENT_BLEND_MODEL_DROPDOWN, CONTENT_BLEND_MORPH_SLIDER ])
	CONTENT_BLEND_MORPH_SLIDER.release(update_content_blend_morph, inputs = CONTENT_BLEND_MORPH_SLIDER)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ CONTENT_BLEND_MODEL_DROPDOWN, CONTENT_BLEND_MORPH_SLIDER ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Slider]:
	has_content_blend = 'content_blend' in processors
	return gradio.Dropdown(visible = has_content_blend), gradio.Slider(visible = has_content_blend and load_processor_module('content_blend').get_inference_pool() and load_processor_module('content_blend').has_morph_input())


def update_content_blend_model(content_blend_model : ContentBlendModel) -> Tuple[gradio.Dropdown, gradio.Slider]:
	content_blend_module = load_processor_module('content_blend')
	content_blend_module.clear_inference_pool()
	state_manager.set_item('content_blend_model', content_blend_model)

	if content_blend_module.pre_check():
		return gradio.Dropdown(value = state_manager.get_item('content_blend_model')), gradio.Slider(visible = content_blend_module.has_morph_input())
	return gradio.Dropdown(), gradio.Slider()


def update_content_blend_morph(content_blend_morph : int) -> None:
	state_manager.set_item('content_blend_morph', content_blend_morph)
