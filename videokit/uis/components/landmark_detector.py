from typing import Optional

import gradio

import videokit.choices
from videokit import landmark_detector, state_manager, translator
from videokit.common_helper import calculate_float_step
from videokit.types import RegionLandmarkerModel, Score
from videokit.uis.core import register_ui_component

landmark_detector_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
landmark_detector_SCORE_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global landmark_detector_MODEL_DROPDOWN
	global landmark_detector_SCORE_SLIDER

	landmark_detector_MODEL_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.landmark_detector_model_dropdown'),
		choices = videokit.choices.landmark_detector_models,
		value = state_manager.get_item('landmark_detector_model')
	)
	landmark_detector_SCORE_SLIDER = gradio.Slider(
		label = translator.get('uis.landmark_detector_score_slider'),
		value = state_manager.get_item('landmark_detector_score'),
		step = calculate_float_step(videokit.choices.landmark_detector_score_range),
		minimum = videokit.choices.landmark_detector_score_range[0],
		maximum = videokit.choices.landmark_detector_score_range[-1]
	)
	register_ui_component('landmark_detector_model_dropdown', landmark_detector_MODEL_DROPDOWN)
	register_ui_component('landmark_detector_score_slider', landmark_detector_SCORE_SLIDER)


def listen() -> None:
	landmark_detector_MODEL_DROPDOWN.change(update_landmark_detector_model, inputs = landmark_detector_MODEL_DROPDOWN, outputs = landmark_detector_MODEL_DROPDOWN)
	landmark_detector_SCORE_SLIDER.release(update_landmark_detector_score, inputs = landmark_detector_SCORE_SLIDER)


def update_landmark_detector_model(landmark_detector_model : RegionLandmarkerModel) -> gradio.Dropdown:
	landmark_detector.clear_inference_pool()
	state_manager.set_item('landmark_detector_model', landmark_detector_model)

	if landmark_detector.pre_check():
		gradio.Dropdown(value = state_manager.get_item('landmark_detector_model'))
	return gradio.Dropdown()


def update_landmark_detector_score(landmark_detector_score : Score) -> None:
	state_manager.set_item('landmark_detector_score', landmark_detector_score)
