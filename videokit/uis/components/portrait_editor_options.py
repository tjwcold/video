from typing import List, Optional, Tuple

import gradio

from videokit import state_manager, translator
from videokit.common_helper import calculate_float_step
from videokit.processors.core import load_processor_module
from videokit.processors.modules.portrait_editor import choices as portrait_editor_choices
from videokit.processors.modules.portrait_editor.types import PortraitEditorModel
from videokit.uis.core import get_ui_component, register_ui_component

PORTRAIT_EDITOR_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
PORTRAIT_EDITOR_EYEBROW_DIRECTION_SLIDER : Optional[gradio.Slider] = None
PORTRAIT_EDITOR_EYE_GAZE_HORIZONTAL_SLIDER : Optional[gradio.Slider] = None
PORTRAIT_EDITOR_EYE_GAZE_VERTICAL_SLIDER : Optional[gradio.Slider] = None
PORTRAIT_EDITOR_EYE_OPEN_RATIO_SLIDER : Optional[gradio.Slider] = None
PORTRAIT_EDITOR_LIP_OPEN_RATIO_SLIDER : Optional[gradio.Slider] = None
PORTRAIT_EDITOR_MOUTH_GRIM_SLIDER : Optional[gradio.Slider] = None
PORTRAIT_EDITOR_MOUTH_POUT_SLIDER : Optional[gradio.Slider] = None
PORTRAIT_EDITOR_MOUTH_PURSE_SLIDER : Optional[gradio.Slider] = None
PORTRAIT_EDITOR_MOUTH_SMILE_SLIDER : Optional[gradio.Slider] = None
PORTRAIT_EDITOR_MOUTH_POSITION_HORIZONTAL_SLIDER : Optional[gradio.Slider] = None
PORTRAIT_EDITOR_MOUTH_POSITION_VERTICAL_SLIDER : Optional[gradio.Slider] = None
PORTRAIT_EDITOR_HEAD_PITCH_SLIDER : Optional[gradio.Slider] = None
PORTRAIT_EDITOR_HEAD_YAW_SLIDER : Optional[gradio.Slider] = None
PORTRAIT_EDITOR_HEAD_ROLL_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global PORTRAIT_EDITOR_MODEL_DROPDOWN
	global PORTRAIT_EDITOR_EYEBROW_DIRECTION_SLIDER
	global PORTRAIT_EDITOR_EYE_GAZE_HORIZONTAL_SLIDER
	global PORTRAIT_EDITOR_EYE_GAZE_VERTICAL_SLIDER
	global PORTRAIT_EDITOR_EYE_OPEN_RATIO_SLIDER
	global PORTRAIT_EDITOR_LIP_OPEN_RATIO_SLIDER
	global PORTRAIT_EDITOR_MOUTH_GRIM_SLIDER
	global PORTRAIT_EDITOR_MOUTH_POUT_SLIDER
	global PORTRAIT_EDITOR_MOUTH_PURSE_SLIDER
	global PORTRAIT_EDITOR_MOUTH_SMILE_SLIDER
	global PORTRAIT_EDITOR_MOUTH_POSITION_HORIZONTAL_SLIDER
	global PORTRAIT_EDITOR_MOUTH_POSITION_VERTICAL_SLIDER
	global PORTRAIT_EDITOR_HEAD_PITCH_SLIDER
	global PORTRAIT_EDITOR_HEAD_YAW_SLIDER
	global PORTRAIT_EDITOR_HEAD_ROLL_SLIDER

	has_portrait_editor = 'portrait_editor' in state_manager.get_item('processors')
	PORTRAIT_EDITOR_MODEL_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.model_dropdown', 'videokit.processors.modules.portrait_editor'),
		choices = portrait_editor_choices.portrait_editor_models,
		value = state_manager.get_item('portrait_editor_model'),
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_EYEBROW_DIRECTION_SLIDER = gradio.Slider(
		label = translator.get('uis.eyebrow_direction_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_eyebrow_direction'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_eyebrow_direction_range),
		minimum = portrait_editor_choices.portrait_editor_eyebrow_direction_range[0],
		maximum = portrait_editor_choices.portrait_editor_eyebrow_direction_range[-1],
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_EYE_GAZE_HORIZONTAL_SLIDER = gradio.Slider(
		label = translator.get('uis.eye_gaze_horizontal_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_eye_gaze_horizontal'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_eye_gaze_horizontal_range),
		minimum = portrait_editor_choices.portrait_editor_eye_gaze_horizontal_range[0],
		maximum = portrait_editor_choices.portrait_editor_eye_gaze_horizontal_range[-1],
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_EYE_GAZE_VERTICAL_SLIDER = gradio.Slider(
		label = translator.get('uis.eye_gaze_vertical_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_eye_gaze_vertical'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_eye_gaze_vertical_range),
		minimum = portrait_editor_choices.portrait_editor_eye_gaze_vertical_range[0],
		maximum = portrait_editor_choices.portrait_editor_eye_gaze_vertical_range[-1],
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_EYE_OPEN_RATIO_SLIDER = gradio.Slider(
		label = translator.get('uis.eye_open_ratio_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_eye_open_ratio'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_eye_open_ratio_range),
		minimum = portrait_editor_choices.portrait_editor_eye_open_ratio_range[0],
		maximum = portrait_editor_choices.portrait_editor_eye_open_ratio_range[-1],
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_LIP_OPEN_RATIO_SLIDER = gradio.Slider(
		label = translator.get('uis.lip_open_ratio_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_lip_open_ratio'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_lip_open_ratio_range),
		minimum = portrait_editor_choices.portrait_editor_lip_open_ratio_range[0],
		maximum = portrait_editor_choices.portrait_editor_lip_open_ratio_range[-1],
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_MOUTH_GRIM_SLIDER = gradio.Slider(
		label = translator.get('uis.mouth_grim_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_mouth_grim'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_mouth_grim_range),
		minimum = portrait_editor_choices.portrait_editor_mouth_grim_range[0],
		maximum = portrait_editor_choices.portrait_editor_mouth_grim_range[-1],
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_MOUTH_POUT_SLIDER = gradio.Slider(
		label = translator.get('uis.mouth_pout_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_mouth_pout'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_mouth_pout_range),
		minimum = portrait_editor_choices.portrait_editor_mouth_pout_range[0],
		maximum = portrait_editor_choices.portrait_editor_mouth_pout_range[-1],
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_MOUTH_PURSE_SLIDER = gradio.Slider(
		label = translator.get('uis.mouth_purse_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_mouth_purse'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_mouth_purse_range),
		minimum = portrait_editor_choices.portrait_editor_mouth_purse_range[0],
		maximum = portrait_editor_choices.portrait_editor_mouth_purse_range[-1],
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_MOUTH_SMILE_SLIDER = gradio.Slider(
		label = translator.get('uis.mouth_smile_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_mouth_smile'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_mouth_smile_range),
		minimum = portrait_editor_choices.portrait_editor_mouth_smile_range[0],
		maximum = portrait_editor_choices.portrait_editor_mouth_smile_range[-1],
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_MOUTH_POSITION_HORIZONTAL_SLIDER = gradio.Slider(
		label = translator.get('uis.mouth_position_horizontal_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_mouth_position_horizontal'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_mouth_position_horizontal_range),
		minimum = portrait_editor_choices.portrait_editor_mouth_position_horizontal_range[0],
		maximum = portrait_editor_choices.portrait_editor_mouth_position_horizontal_range[-1],
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_MOUTH_POSITION_VERTICAL_SLIDER = gradio.Slider(
		label = translator.get('uis.mouth_position_vertical_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_mouth_position_vertical'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_mouth_position_vertical_range),
		minimum = portrait_editor_choices.portrait_editor_mouth_position_vertical_range[0],
		maximum = portrait_editor_choices.portrait_editor_mouth_position_vertical_range[-1],
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_HEAD_PITCH_SLIDER = gradio.Slider(
		label = translator.get('uis.head_pitch_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_head_pitch'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_head_pitch_range),
		minimum = portrait_editor_choices.portrait_editor_head_pitch_range[0],
		maximum = portrait_editor_choices.portrait_editor_head_pitch_range[-1],
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_HEAD_YAW_SLIDER = gradio.Slider(
		label = translator.get('uis.head_yaw_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_head_yaw'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_head_yaw_range),
		minimum = portrait_editor_choices.portrait_editor_head_yaw_range[0],
		maximum = portrait_editor_choices.portrait_editor_head_yaw_range[-1],
		visible = has_portrait_editor
	)
	PORTRAIT_EDITOR_HEAD_ROLL_SLIDER = gradio.Slider(
		label = translator.get('uis.head_roll_slider', 'videokit.processors.modules.portrait_editor'),
		value = state_manager.get_item('portrait_editor_head_roll'),
		step = calculate_float_step(portrait_editor_choices.portrait_editor_head_roll_range),
		minimum = portrait_editor_choices.portrait_editor_head_roll_range[0],
		maximum = portrait_editor_choices.portrait_editor_head_roll_range[-1],
		visible = has_portrait_editor
	)
	register_ui_component('portrait_editor_model_dropdown', PORTRAIT_EDITOR_MODEL_DROPDOWN)
	register_ui_component('portrait_editor_eyebrow_direction_slider', PORTRAIT_EDITOR_EYEBROW_DIRECTION_SLIDER)
	register_ui_component('portrait_editor_eye_gaze_horizontal_slider', PORTRAIT_EDITOR_EYE_GAZE_HORIZONTAL_SLIDER)
	register_ui_component('portrait_editor_eye_gaze_vertical_slider', PORTRAIT_EDITOR_EYE_GAZE_VERTICAL_SLIDER)
	register_ui_component('portrait_editor_eye_open_ratio_slider', PORTRAIT_EDITOR_EYE_OPEN_RATIO_SLIDER)
	register_ui_component('portrait_editor_lip_open_ratio_slider', PORTRAIT_EDITOR_LIP_OPEN_RATIO_SLIDER)
	register_ui_component('portrait_editor_mouth_grim_slider', PORTRAIT_EDITOR_MOUTH_GRIM_SLIDER)
	register_ui_component('portrait_editor_mouth_pout_slider', PORTRAIT_EDITOR_MOUTH_POUT_SLIDER)
	register_ui_component('portrait_editor_mouth_purse_slider', PORTRAIT_EDITOR_MOUTH_PURSE_SLIDER)
	register_ui_component('portrait_editor_mouth_smile_slider', PORTRAIT_EDITOR_MOUTH_SMILE_SLIDER)
	register_ui_component('portrait_editor_mouth_position_horizontal_slider', PORTRAIT_EDITOR_MOUTH_POSITION_HORIZONTAL_SLIDER)
	register_ui_component('portrait_editor_mouth_position_vertical_slider', PORTRAIT_EDITOR_MOUTH_POSITION_VERTICAL_SLIDER)
	register_ui_component('portrait_editor_head_pitch_slider', PORTRAIT_EDITOR_HEAD_PITCH_SLIDER)
	register_ui_component('portrait_editor_head_yaw_slider', PORTRAIT_EDITOR_HEAD_YAW_SLIDER)
	register_ui_component('portrait_editor_head_roll_slider', PORTRAIT_EDITOR_HEAD_ROLL_SLIDER)


def listen() -> None:
	PORTRAIT_EDITOR_MODEL_DROPDOWN.change(update_portrait_editor_model, inputs = PORTRAIT_EDITOR_MODEL_DROPDOWN, outputs = PORTRAIT_EDITOR_MODEL_DROPDOWN)
	PORTRAIT_EDITOR_EYEBROW_DIRECTION_SLIDER.release(update_portrait_editor_eyebrow_direction, inputs = PORTRAIT_EDITOR_EYEBROW_DIRECTION_SLIDER)
	PORTRAIT_EDITOR_EYE_GAZE_HORIZONTAL_SLIDER.release(update_portrait_editor_eye_gaze_horizontal, inputs = PORTRAIT_EDITOR_EYE_GAZE_HORIZONTAL_SLIDER)
	PORTRAIT_EDITOR_EYE_GAZE_VERTICAL_SLIDER.release(update_portrait_editor_eye_gaze_vertical, inputs = PORTRAIT_EDITOR_EYE_GAZE_VERTICAL_SLIDER)
	PORTRAIT_EDITOR_EYE_OPEN_RATIO_SLIDER.release(update_portrait_editor_eye_open_ratio, inputs = PORTRAIT_EDITOR_EYE_OPEN_RATIO_SLIDER)
	PORTRAIT_EDITOR_LIP_OPEN_RATIO_SLIDER.release(update_portrait_editor_lip_open_ratio, inputs = PORTRAIT_EDITOR_LIP_OPEN_RATIO_SLIDER)
	PORTRAIT_EDITOR_MOUTH_GRIM_SLIDER.release(update_portrait_editor_mouth_grim, inputs = PORTRAIT_EDITOR_MOUTH_GRIM_SLIDER)
	PORTRAIT_EDITOR_MOUTH_POUT_SLIDER.release(update_portrait_editor_mouth_pout, inputs = PORTRAIT_EDITOR_MOUTH_POUT_SLIDER)
	PORTRAIT_EDITOR_MOUTH_PURSE_SLIDER.release(update_portrait_editor_mouth_purse, inputs = PORTRAIT_EDITOR_MOUTH_PURSE_SLIDER)
	PORTRAIT_EDITOR_MOUTH_SMILE_SLIDER.release(update_portrait_editor_mouth_smile, inputs = PORTRAIT_EDITOR_MOUTH_SMILE_SLIDER)
	PORTRAIT_EDITOR_MOUTH_POSITION_HORIZONTAL_SLIDER.release(update_portrait_editor_mouth_position_horizontal, inputs = PORTRAIT_EDITOR_MOUTH_POSITION_HORIZONTAL_SLIDER)
	PORTRAIT_EDITOR_MOUTH_POSITION_VERTICAL_SLIDER.release(update_portrait_editor_mouth_position_vertical, inputs = PORTRAIT_EDITOR_MOUTH_POSITION_VERTICAL_SLIDER)
	PORTRAIT_EDITOR_HEAD_PITCH_SLIDER.release(update_portrait_editor_head_pitch, inputs = PORTRAIT_EDITOR_HEAD_PITCH_SLIDER)
	PORTRAIT_EDITOR_HEAD_YAW_SLIDER.release(update_portrait_editor_head_yaw, inputs = PORTRAIT_EDITOR_HEAD_YAW_SLIDER)
	PORTRAIT_EDITOR_HEAD_ROLL_SLIDER.release(update_portrait_editor_head_roll, inputs = PORTRAIT_EDITOR_HEAD_ROLL_SLIDER)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ PORTRAIT_EDITOR_MODEL_DROPDOWN, PORTRAIT_EDITOR_EYEBROW_DIRECTION_SLIDER, PORTRAIT_EDITOR_EYE_GAZE_HORIZONTAL_SLIDER, PORTRAIT_EDITOR_EYE_GAZE_VERTICAL_SLIDER, PORTRAIT_EDITOR_EYE_OPEN_RATIO_SLIDER, PORTRAIT_EDITOR_LIP_OPEN_RATIO_SLIDER, PORTRAIT_EDITOR_MOUTH_GRIM_SLIDER, PORTRAIT_EDITOR_MOUTH_POUT_SLIDER, PORTRAIT_EDITOR_MOUTH_PURSE_SLIDER, PORTRAIT_EDITOR_MOUTH_SMILE_SLIDER, PORTRAIT_EDITOR_MOUTH_POSITION_HORIZONTAL_SLIDER, PORTRAIT_EDITOR_MOUTH_POSITION_VERTICAL_SLIDER, PORTRAIT_EDITOR_HEAD_PITCH_SLIDER, PORTRAIT_EDITOR_HEAD_YAW_SLIDER, PORTRAIT_EDITOR_HEAD_ROLL_SLIDER ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider]:
	has_portrait_editor = 'portrait_editor' in processors
	return gradio.Dropdown(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor), gradio.Slider(visible = has_portrait_editor)


def update_portrait_editor_model(portrait_editor_model : PortraitEditorModel) -> gradio.Dropdown:
	portrait_editor_module = load_processor_module('portrait_editor')
	portrait_editor_module.clear_inference_pool()
	state_manager.set_item('portrait_editor_model', portrait_editor_model)

	if portrait_editor_module.pre_check():
		return gradio.Dropdown(value = state_manager.get_item('portrait_editor_model'))
	return gradio.Dropdown()


def update_portrait_editor_eyebrow_direction(portrait_editor_eyebrow_direction : float) -> None:
	state_manager.set_item('portrait_editor_eyebrow_direction', portrait_editor_eyebrow_direction)


def update_portrait_editor_eye_gaze_horizontal(portrait_editor_eye_gaze_horizontal : float) -> None:
	state_manager.set_item('portrait_editor_eye_gaze_horizontal', portrait_editor_eye_gaze_horizontal)


def update_portrait_editor_eye_gaze_vertical(portrait_editor_eye_gaze_vertical : float) -> None:
	state_manager.set_item('portrait_editor_eye_gaze_vertical', portrait_editor_eye_gaze_vertical)


def update_portrait_editor_eye_open_ratio(portrait_editor_eye_open_ratio : float) -> None:
	state_manager.set_item('portrait_editor_eye_open_ratio', portrait_editor_eye_open_ratio)


def update_portrait_editor_lip_open_ratio(portrait_editor_lip_open_ratio : float) -> None:
	state_manager.set_item('portrait_editor_lip_open_ratio', portrait_editor_lip_open_ratio)


def update_portrait_editor_mouth_grim(portrait_editor_mouth_grim : float) -> None:
	state_manager.set_item('portrait_editor_mouth_grim', portrait_editor_mouth_grim)


def update_portrait_editor_mouth_pout(portrait_editor_mouth_pout : float) -> None:
	state_manager.set_item('portrait_editor_mouth_pout', portrait_editor_mouth_pout)


def update_portrait_editor_mouth_purse(portrait_editor_mouth_purse : float) -> None:
	state_manager.set_item('portrait_editor_mouth_purse', portrait_editor_mouth_purse)


def update_portrait_editor_mouth_smile(portrait_editor_mouth_smile : float) -> None:
	state_manager.set_item('portrait_editor_mouth_smile', portrait_editor_mouth_smile)


def update_portrait_editor_mouth_position_horizontal(portrait_editor_mouth_position_horizontal : float) -> None:
	state_manager.set_item('portrait_editor_mouth_position_horizontal', portrait_editor_mouth_position_horizontal)


def update_portrait_editor_mouth_position_vertical(portrait_editor_mouth_position_vertical : float) -> None:
	state_manager.set_item('portrait_editor_mouth_position_vertical', portrait_editor_mouth_position_vertical)


def update_portrait_editor_head_pitch(portrait_editor_head_pitch : float) -> None:
	state_manager.set_item('portrait_editor_head_pitch', portrait_editor_head_pitch)


def update_portrait_editor_head_yaw(portrait_editor_head_yaw : float) -> None:
	state_manager.set_item('portrait_editor_head_yaw', portrait_editor_head_yaw)


def update_portrait_editor_head_roll(portrait_editor_head_roll : float) -> None:
	state_manager.set_item('portrait_editor_head_roll', portrait_editor_head_roll)
