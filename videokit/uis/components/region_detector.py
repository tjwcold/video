from typing import Optional, Sequence, Tuple

import gradio

import videokit.choices
from videokit import region_detector, state_manager, translator
from videokit.common_helper import calculate_float_step, get_last
from videokit.sanitizer import sanitize_int_range
from videokit.types import Angle, FaceDetectorModel, Score
from videokit.uis.core import register_ui_component
from videokit.uis.types import ComponentOptions

region_detector_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
region_detector_SIZE_DROPDOWN : Optional[gradio.Dropdown] = None
region_detector_MARGIN_SLIDER : Optional[gradio.Slider] = None
region_detector_ANGLES_CHECKBOX_GROUP : Optional[gradio.CheckboxGroup] = None
region_detector_SCORE_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global region_detector_MODEL_DROPDOWN
	global region_detector_SIZE_DROPDOWN
	global region_detector_MARGIN_SLIDER
	global region_detector_ANGLES_CHECKBOX_GROUP
	global region_detector_SCORE_SLIDER

	region_detector_size_dropdown_options : ComponentOptions =\
	{
		'label': translator.get('uis.region_detector_size_dropdown'),
		'value': state_manager.get_item('region_detector_size')
	}
	if state_manager.get_item('region_detector_size') in videokit.choices.region_detector_set[state_manager.get_item('region_detector_model')]:
		region_detector_size_dropdown_options['choices'] = videokit.choices.region_detector_set[state_manager.get_item('region_detector_model')]
	with gradio.Row():
		region_detector_MODEL_DROPDOWN = gradio.Dropdown(
			label = translator.get('uis.region_detector_model_dropdown'),
			choices = videokit.choices.region_detector_models,
			value = state_manager.get_item('region_detector_model')
		)
		region_detector_SIZE_DROPDOWN = gradio.Dropdown(**region_detector_size_dropdown_options)
	region_detector_MARGIN_SLIDER = gradio.Slider(
		label = translator.get('uis.region_detector_margin_slider'),
		value = state_manager.get_item('region_detector_margin')[0],
		step = calculate_float_step(videokit.choices.region_detector_margin_range),
		minimum = videokit.choices.region_detector_margin_range[0],
		maximum = videokit.choices.region_detector_margin_range[-1]
	)
	region_detector_ANGLES_CHECKBOX_GROUP = gradio.CheckboxGroup(
		label = translator.get('uis.region_detector_angles_checkbox_group'),
		choices = videokit.choices.region_detector_angles,
		value = state_manager.get_item('region_detector_angles')
	)
	region_detector_SCORE_SLIDER = gradio.Slider(
		label = translator.get('uis.region_detector_score_slider'),
		value = state_manager.get_item('region_detector_score'),
		step = calculate_float_step(videokit.choices.region_detector_score_range),
		minimum = videokit.choices.region_detector_score_range[0],
		maximum = videokit.choices.region_detector_score_range[-1]
	)
	register_ui_component('region_detector_model_dropdown', region_detector_MODEL_DROPDOWN)
	register_ui_component('region_detector_size_dropdown', region_detector_SIZE_DROPDOWN)
	register_ui_component('region_detector_margin_slider', region_detector_MARGIN_SLIDER)
	register_ui_component('region_detector_angles_checkbox_group', region_detector_ANGLES_CHECKBOX_GROUP)
	register_ui_component('region_detector_score_slider', region_detector_SCORE_SLIDER)


def listen() -> None:
	region_detector_MODEL_DROPDOWN.change(update_region_detector_model, inputs = region_detector_MODEL_DROPDOWN, outputs = [ region_detector_MODEL_DROPDOWN, region_detector_SIZE_DROPDOWN ])
	region_detector_SIZE_DROPDOWN.change(update_region_detector_size, inputs = region_detector_SIZE_DROPDOWN)
	region_detector_MARGIN_SLIDER.release(update_region_detector_margin, inputs=region_detector_MARGIN_SLIDER)
	region_detector_ANGLES_CHECKBOX_GROUP.change(update_region_detector_angles, inputs = region_detector_ANGLES_CHECKBOX_GROUP, outputs = region_detector_ANGLES_CHECKBOX_GROUP)
	region_detector_SCORE_SLIDER.release(update_region_detector_score, inputs = region_detector_SCORE_SLIDER)


def update_region_detector_model(region_detector_model : FaceDetectorModel) -> Tuple[gradio.Dropdown, gradio.Dropdown]:
	region_detector.clear_inference_pool()
	state_manager.set_item('region_detector_model', region_detector_model)

	if region_detector.pre_check():
		region_detector_size_choices = videokit.choices.region_detector_set.get(state_manager.get_item('region_detector_model'))
		state_manager.set_item('region_detector_size', get_last(region_detector_size_choices))
		return gradio.Dropdown(value = state_manager.get_item('region_detector_model')), gradio.Dropdown(value = state_manager.get_item('region_detector_size'), choices = region_detector_size_choices)
	return gradio.Dropdown(), gradio.Dropdown()


def update_region_detector_size(region_detector_size : str) -> None:
	state_manager.set_item('region_detector_size', region_detector_size)


def update_region_detector_margin(region_detector_margin : int) -> None:
	region_detector_margin = sanitize_int_range(region_detector_margin, videokit.choices.region_detector_margin_range)
	state_manager.set_item('region_detector_margin', (region_detector_margin, region_detector_margin, region_detector_margin, region_detector_margin))


def update_region_detector_angles(region_detector_angles : Sequence[Angle]) -> gradio.CheckboxGroup:
	region_detector_angles = region_detector_angles or videokit.choices.region_detector_angles

	state_manager.set_item('region_detector_angles', region_detector_angles)
	return gradio.CheckboxGroup(value = state_manager.get_item('region_detector_angles'))


def update_region_detector_score(region_detector_score : Score) -> None:
	state_manager.set_item('region_detector_score', region_detector_score)
