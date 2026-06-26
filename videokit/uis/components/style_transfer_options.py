from typing import List, Optional, Tuple

import gradio

from videokit import state_manager, translator
from videokit.common_helper import calculate_float_step, get_first
from videokit.processors.core import load_processor_module
from videokit.processors.modules.style_transfer import choices as style_transfer_choices
from videokit.processors.modules.style_transfer.types import StyleTransferModel, StyleTransferWeight
from videokit.uis.core import get_ui_component, register_ui_component

STYLE_TRANSFER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
STYLE_TRANSFER_PIXEL_BOOST_DROPDOWN : Optional[gradio.Dropdown] = None
STYLE_TRANSFER_WEIGHT_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global STYLE_TRANSFER_MODEL_DROPDOWN
	global STYLE_TRANSFER_PIXEL_BOOST_DROPDOWN
	global STYLE_TRANSFER_WEIGHT_SLIDER

	has_style_transfer = 'style_transfer' in state_manager.get_item('processors')
	STYLE_TRANSFER_MODEL_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.model_dropdown', 'videokit.processors.modules.style_transfer'),
		choices = style_transfer_choices.style_transfer_models,
		value = state_manager.get_item('style_transfer_model'),
		visible = has_style_transfer
	)
	STYLE_TRANSFER_PIXEL_BOOST_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.pixel_boost_dropdown', 'videokit.processors.modules.style_transfer'),
		choices = style_transfer_choices.style_transfer_set.get(state_manager.get_item('style_transfer_model')),
		value = state_manager.get_item('style_transfer_pixel_boost'),
		visible = has_style_transfer
	)
	STYLE_TRANSFER_WEIGHT_SLIDER = gradio.Slider(
		label = translator.get('uis.weight_slider', 'videokit.processors.modules.style_transfer'),
		value = state_manager.get_item('style_transfer_weight'),
		minimum = style_transfer_choices.style_transfer_weight_range[0],
		maximum = style_transfer_choices.style_transfer_weight_range[-1],
		step = calculate_float_step(style_transfer_choices.style_transfer_weight_range),
		visible = has_style_transfer and has_style_transfer_weight()
	)
	register_ui_component('style_transfer_model_dropdown', STYLE_TRANSFER_MODEL_DROPDOWN)
	register_ui_component('style_transfer_pixel_boost_dropdown', STYLE_TRANSFER_PIXEL_BOOST_DROPDOWN)
	register_ui_component('style_transfer_weight_slider', STYLE_TRANSFER_WEIGHT_SLIDER)


def listen() -> None:
	STYLE_TRANSFER_MODEL_DROPDOWN.change(update_style_transfer_model, inputs = STYLE_TRANSFER_MODEL_DROPDOWN, outputs = [ STYLE_TRANSFER_MODEL_DROPDOWN, STYLE_TRANSFER_PIXEL_BOOST_DROPDOWN, STYLE_TRANSFER_WEIGHT_SLIDER ])
	STYLE_TRANSFER_PIXEL_BOOST_DROPDOWN.change(update_style_transfer_pixel_boost, inputs = STYLE_TRANSFER_PIXEL_BOOST_DROPDOWN)
	STYLE_TRANSFER_WEIGHT_SLIDER.change(update_style_transfer_weight, inputs = STYLE_TRANSFER_WEIGHT_SLIDER)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ STYLE_TRANSFER_MODEL_DROPDOWN, STYLE_TRANSFER_PIXEL_BOOST_DROPDOWN, STYLE_TRANSFER_WEIGHT_SLIDER ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Dropdown, gradio.Slider]:
	has_style_transfer = 'style_transfer' in processors
	return gradio.Dropdown(visible = has_style_transfer), gradio.Dropdown(visible = has_style_transfer), gradio.Slider(visible = has_style_transfer)


def update_style_transfer_model(style_transfer_model : StyleTransferModel) -> Tuple[gradio.Dropdown, gradio.Dropdown, gradio.Slider]:
	style_transfer_module = load_processor_module('style_transfer')
	style_transfer_module.clear_inference_pool()
	state_manager.set_item('style_transfer_model', style_transfer_model)

	if style_transfer_module.pre_check():
		style_transfer_pixel_boost_dropdown_choices = style_transfer_choices.style_transfer_set.get(state_manager.get_item('style_transfer_model'))
		state_manager.set_item('style_transfer_pixel_boost', get_first(style_transfer_pixel_boost_dropdown_choices))
		return gradio.Dropdown(value = state_manager.get_item('style_transfer_model')), gradio.Dropdown(value = state_manager.get_item('style_transfer_pixel_boost'), choices = style_transfer_pixel_boost_dropdown_choices), gradio.Slider(visible = has_style_transfer_weight())
	return gradio.Dropdown(), gradio.Dropdown(), gradio.Slider()


def update_style_transfer_pixel_boost(style_transfer_pixel_boost : str) -> None:
	state_manager.set_item('style_transfer_pixel_boost', style_transfer_pixel_boost)


def update_style_transfer_weight(style_transfer_weight : StyleTransferWeight) -> None:
	state_manager.set_item('style_transfer_weight', style_transfer_weight)


def has_style_transfer_weight() -> bool:
	return state_manager.get_item('style_transfer_model') in [ 'ghost_1_256', 'ghost_2_256', 'ghost_3_256', 'hififace_unofficial_256', 'hyperswap_1a_256', 'hyperswap_1b_256', 'hyperswap_1c_256', 'inswapper_128', 'inswapper_128_fp16', 'simswap_256', 'simswap_unofficial_512' ]
