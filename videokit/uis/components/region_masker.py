from typing import List, Optional, Tuple

import gradio

import videokit.choices
from videokit import region_masker, state_manager, translator
from videokit.common_helper import calculate_float_step, calculate_int_step
from videokit.sanitizer import sanitize_int_range
from videokit.types import RegionMaskArea, RegionMaskRegion, RegionMaskType, RegionOccluderModel, RegionParserModel
from videokit.uis.core import register_ui_component

REGION_OCCLUDER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
REGION_PARSER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
REGION_MASK_TYPES_CHECKBOX_GROUP : Optional[gradio.CheckboxGroup] = None
REGION_MASK_AREAS_CHECKBOX_GROUP : Optional[gradio.CheckboxGroup] = None
REGION_MASK_REGIONS_CHECKBOX_GROUP : Optional[gradio.CheckboxGroup] = None
REGION_MASK_BOX_WRAPPER : Optional[gradio.Group] = None
REGION_MASK_BLUR_SLIDER : Optional[gradio.Slider] = None
REGION_MASK_PADDING_TOP_SLIDER : Optional[gradio.Slider] = None
REGION_MASK_PADDING_RIGHT_SLIDER : Optional[gradio.Slider] = None
REGION_MASK_PADDING_BOTTOM_SLIDER : Optional[gradio.Slider] = None
REGION_MASK_PADDING_LEFT_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global REGION_OCCLUDER_MODEL_DROPDOWN
	global REGION_PARSER_MODEL_DROPDOWN
	global REGION_MASK_TYPES_CHECKBOX_GROUP
	global REGION_MASK_AREAS_CHECKBOX_GROUP
	global REGION_MASK_REGIONS_CHECKBOX_GROUP
	global REGION_MASK_BOX_WRAPPER
	global REGION_MASK_BLUR_SLIDER
	global REGION_MASK_PADDING_TOP_SLIDER
	global REGION_MASK_PADDING_RIGHT_SLIDER
	global REGION_MASK_PADDING_BOTTOM_SLIDER
	global REGION_MASK_PADDING_LEFT_SLIDER

	has_box_mask = 'box' in state_manager.get_item('region_mask_types')
	has_region_mask = 'region' in state_manager.get_item('region_mask_types')
	has_area_mask = 'area' in state_manager.get_item('region_mask_types')
	with gradio.Row():
		REGION_OCCLUDER_MODEL_DROPDOWN = gradio.Dropdown(
			label = translator.get('uis.region_occluder_model_dropdown'),
			choices = videokit.choices.region_occluder_models,
			value = state_manager.get_item('region_occluder_model')
		)
		REGION_PARSER_MODEL_DROPDOWN = gradio.Dropdown(
			label = translator.get('uis.region_parser_model_dropdown'),
			choices = videokit.choices.region_parser_models,
			value = state_manager.get_item('region_parser_model')
		)
	REGION_MASK_TYPES_CHECKBOX_GROUP = gradio.CheckboxGroup(
		label = translator.get('uis.region_mask_types_checkbox_group'),
		choices = videokit.choices.region_mask_types,
		value = state_manager.get_item('region_mask_types')
	)
	REGION_MASK_AREAS_CHECKBOX_GROUP = gradio.CheckboxGroup(
		label = translator.get('uis.region_mask_areas_checkbox_group'),
		choices = videokit.choices.region_mask_areas,
		value = state_manager.get_item('region_mask_areas'),
		visible = has_area_mask
	)
	REGION_MASK_REGIONS_CHECKBOX_GROUP = gradio.CheckboxGroup(
		label = translator.get('uis.region_mask_regions_checkbox_group'),
		choices = videokit.choices.region_mask_regions,
		value = state_manager.get_item('region_mask_regions'),
		visible = has_region_mask
	)
	REGION_MASK_BLUR_SLIDER = gradio.Slider(
		label = translator.get('uis.region_mask_blur_slider'),
		step = calculate_float_step(videokit.choices.region_mask_blur_range),
		minimum = videokit.choices.region_mask_blur_range[0],
		maximum = videokit.choices.region_mask_blur_range[-1],
		value = state_manager.get_item('region_mask_blur'),
		visible = has_box_mask
	)
	with gradio.Group(visible = has_box_mask) as REGION_MASK_BOX_WRAPPER:
		with gradio.Row():
			REGION_MASK_PADDING_TOP_SLIDER = gradio.Slider(
				label = translator.get('uis.region_mask_padding_top_slider'),
				step = calculate_int_step(videokit.choices.region_mask_padding_range),
				minimum = videokit.choices.region_mask_padding_range[0],
				maximum = videokit.choices.region_mask_padding_range[-1],
				value = state_manager.get_item('region_mask_padding')[0]
			)
			REGION_MASK_PADDING_RIGHT_SLIDER = gradio.Slider(
				label = translator.get('uis.region_mask_padding_right_slider'),
				step = calculate_int_step(videokit.choices.region_mask_padding_range),
				minimum = videokit.choices.region_mask_padding_range[0],
				maximum = videokit.choices.region_mask_padding_range[-1],
				value = state_manager.get_item('region_mask_padding')[1]
			)
		with gradio.Row():
			REGION_MASK_PADDING_BOTTOM_SLIDER = gradio.Slider(
				label = translator.get('uis.region_mask_padding_bottom_slider'),
				step = calculate_int_step(videokit.choices.region_mask_padding_range),
				minimum = videokit.choices.region_mask_padding_range[0],
				maximum = videokit.choices.region_mask_padding_range[-1],
				value = state_manager.get_item('region_mask_padding')[2]
			)
			REGION_MASK_PADDING_LEFT_SLIDER = gradio.Slider(
				label = translator.get('uis.region_mask_padding_left_slider'),
				step = calculate_int_step(videokit.choices.region_mask_padding_range),
				minimum = videokit.choices.region_mask_padding_range[0],
				maximum = videokit.choices.region_mask_padding_range[-1],
				value = state_manager.get_item('region_mask_padding')[3]
			)
	register_ui_component('region_occluder_model_dropdown', REGION_OCCLUDER_MODEL_DROPDOWN)
	register_ui_component('region_parser_model_dropdown', REGION_PARSER_MODEL_DROPDOWN)
	register_ui_component('region_mask_types_checkbox_group', REGION_MASK_TYPES_CHECKBOX_GROUP)
	register_ui_component('region_mask_areas_checkbox_group', REGION_MASK_AREAS_CHECKBOX_GROUP)
	register_ui_component('region_mask_regions_checkbox_group', REGION_MASK_REGIONS_CHECKBOX_GROUP)
	register_ui_component('region_mask_blur_slider', REGION_MASK_BLUR_SLIDER)
	register_ui_component('region_mask_padding_top_slider', REGION_MASK_PADDING_TOP_SLIDER)
	register_ui_component('region_mask_padding_right_slider', REGION_MASK_PADDING_RIGHT_SLIDER)
	register_ui_component('region_mask_padding_bottom_slider', REGION_MASK_PADDING_BOTTOM_SLIDER)
	register_ui_component('region_mask_padding_left_slider', REGION_MASK_PADDING_LEFT_SLIDER)


def listen() -> None:
	REGION_OCCLUDER_MODEL_DROPDOWN.change(update_region_occluder_model, inputs = REGION_OCCLUDER_MODEL_DROPDOWN)
	REGION_PARSER_MODEL_DROPDOWN.change(update_region_parser_model, inputs = REGION_PARSER_MODEL_DROPDOWN)
	REGION_MASK_TYPES_CHECKBOX_GROUP.change(update_region_mask_types, inputs = REGION_MASK_TYPES_CHECKBOX_GROUP, outputs = [ REGION_MASK_TYPES_CHECKBOX_GROUP, REGION_MASK_AREAS_CHECKBOX_GROUP, REGION_MASK_REGIONS_CHECKBOX_GROUP, REGION_MASK_BLUR_SLIDER, REGION_MASK_BOX_WRAPPER ])
	REGION_MASK_AREAS_CHECKBOX_GROUP.change(update_region_mask_areas, inputs = REGION_MASK_AREAS_CHECKBOX_GROUP, outputs = REGION_MASK_AREAS_CHECKBOX_GROUP)
	REGION_MASK_REGIONS_CHECKBOX_GROUP.change(update_region_mask_regions, inputs = REGION_MASK_REGIONS_CHECKBOX_GROUP, outputs = REGION_MASK_REGIONS_CHECKBOX_GROUP)
	REGION_MASK_BLUR_SLIDER.release(update_region_mask_blur, inputs = REGION_MASK_BLUR_SLIDER)

	region_mask_padding_sliders = [ REGION_MASK_PADDING_TOP_SLIDER, REGION_MASK_PADDING_RIGHT_SLIDER, REGION_MASK_PADDING_BOTTOM_SLIDER, REGION_MASK_PADDING_LEFT_SLIDER ]
	for region_mask_padding_slider in region_mask_padding_sliders:
		region_mask_padding_slider.release(update_region_mask_padding, inputs = region_mask_padding_sliders)


def update_region_occluder_model(region_occluder_model : RegionOccluderModel) -> gradio.Dropdown:
	region_masker.clear_inference_pool()
	state_manager.set_item('region_occluder_model', region_occluder_model)

	if region_masker.pre_check():
		return gradio.Dropdown(value = state_manager.get_item('region_occluder_model'))
	return gradio.Dropdown()


def update_region_parser_model(region_parser_model : RegionParserModel) -> gradio.Dropdown:
	region_masker.clear_inference_pool()
	state_manager.set_item('region_parser_model', region_parser_model)

	if region_masker.pre_check():
		return gradio.Dropdown(value = state_manager.get_item('region_parser_model'))
	return gradio.Dropdown()


def update_region_mask_types(region_mask_types : List[RegionMaskType]) -> Tuple[gradio.CheckboxGroup, gradio.CheckboxGroup, gradio.CheckboxGroup, gradio.Slider, gradio.Group]:
	region_mask_types = region_mask_types or videokit.choices.region_mask_types
	state_manager.set_item('region_mask_types', region_mask_types)
	has_box_mask = 'box' in region_mask_types
	has_area_mask = 'area' in region_mask_types
	has_region_mask = 'region' in region_mask_types
	return gradio.CheckboxGroup(value = state_manager.get_item('region_mask_types')), gradio.CheckboxGroup(visible = has_area_mask), gradio.CheckboxGroup(visible = has_region_mask), gradio.Slider(visible = has_box_mask), gradio.Group(visible = has_box_mask)


def update_region_mask_areas(region_mask_areas : List[RegionMaskArea]) -> gradio.CheckboxGroup:
	region_mask_areas = region_mask_areas or videokit.choices.region_mask_areas
	state_manager.set_item('region_mask_areas', region_mask_areas)
	return gradio.CheckboxGroup(value = state_manager.get_item('region_mask_areas'))


def update_region_mask_regions(region_mask_regions : List[RegionMaskRegion]) -> gradio.CheckboxGroup:
	region_mask_regions = region_mask_regions or videokit.choices.region_mask_regions
	state_manager.set_item('region_mask_regions', region_mask_regions)
	return gradio.CheckboxGroup(value = state_manager.get_item('region_mask_regions'))


def update_region_mask_blur(region_mask_blur : float) -> None:
	state_manager.set_item('region_mask_blur', region_mask_blur)


def update_region_mask_padding(region_mask_padding_top : float, region_mask_padding_right : float, region_mask_padding_bottom : float, region_mask_padding_left : float) -> None:
	region_mask_padding_top = sanitize_int_range(int(region_mask_padding_top), videokit.choices.region_mask_padding_range)
	region_mask_padding_right = sanitize_int_range(int(region_mask_padding_right), videokit.choices.region_mask_padding_range)
	region_mask_padding_bottom = sanitize_int_range(int(region_mask_padding_bottom), videokit.choices.region_mask_padding_range)
	region_mask_padding_left = sanitize_int_range(int(region_mask_padding_left), videokit.choices.region_mask_padding_range)
	state_manager.set_item('region_mask_padding', (region_mask_padding_top, region_mask_padding_right, region_mask_padding_bottom, region_mask_padding_left))
