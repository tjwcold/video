from typing import List, Optional, Tuple

import cv2
import gradio
from gradio_rangeslider import RangeSlider

import videokit.choices
from videokit import state_manager, translator
from videokit.common_helper import calculate_float_step, calculate_int_step
from videokit.region_analyser import get_many_regions
from videokit.region_selector import sort_and_filter_regions
from videokit.region_store import clear_static_regions
from videokit.filesystem import is_image, is_video
from videokit.types import RegionSelectorMode, RegionSelectorOrder, Gender, Race, VisionFrame
from videokit.uis.core import get_ui_component, get_ui_components, register_ui_component
from videokit.uis.types import ComponentOptions
from videokit.uis.ui_helper import convert_str_none
from videokit.vision import fit_cover_frame, read_static_image, read_video_frame

REGION_SELECTOR_MODE_DROPDOWN : Optional[gradio.Dropdown] = None
REGION_SELECTOR_ORDER_DROPDOWN : Optional[gradio.Dropdown] = None
REGION_SELECTOR_GENDER_DROPDOWN : Optional[gradio.Dropdown] = None
REGION_SELECTOR_RACE_DROPDOWN : Optional[gradio.Dropdown] = None
REGION_SELECTOR_AGE_RANGE_SLIDER : Optional[RangeSlider] = None
REFERENCE_REGION_POSITION_GALLERY : Optional[gradio.Gallery] = None
REFERENCE_REGION_DISTANCE_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global REGION_SELECTOR_MODE_DROPDOWN
	global REGION_SELECTOR_ORDER_DROPDOWN
	global REGION_SELECTOR_GENDER_DROPDOWN
	global REGION_SELECTOR_RACE_DROPDOWN
	global REGION_SELECTOR_AGE_RANGE_SLIDER
	global REFERENCE_REGION_POSITION_GALLERY
	global REFERENCE_REGION_DISTANCE_SLIDER

	reference_region_gallery_options : ComponentOptions =\
	{
		'label': translator.get('uis.reference_region_gallery'),
		'object_fit': 'cover',
		'columns': 7,
		'allow_preview': False,
		'elem_classes': 'box-region-selector',
		'visible': 'reference' in state_manager.get_item('region_selector_mode')
	}
	if is_image(state_manager.get_item('target_path')):
		target_vision_frame = read_static_image(state_manager.get_item('target_path'))
		reference_region_gallery_options['value'] = extract_gallery_frames(target_vision_frame)
	if is_video(state_manager.get_item('target_path')):
		target_vision_frame = read_video_frame(state_manager.get_item('target_path'), state_manager.get_item('reference_frame_number'))
		reference_region_gallery_options['value'] = extract_gallery_frames(target_vision_frame)
	REGION_SELECTOR_MODE_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.region_selector_mode_dropdown'),
		choices = videokit.choices.region_selector_modes,
		value = state_manager.get_item('region_selector_mode')
	)
	REFERENCE_REGION_POSITION_GALLERY = gradio.Gallery(**reference_region_gallery_options)
	with gradio.Group():
		with gradio.Row():
			REGION_SELECTOR_ORDER_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.region_selector_order_dropdown'),
		choices = videokit.choices.region_selector_orders,
		value = state_manager.get_item('region_selector_order')
	)
	REGION_SELECTOR_GENDER_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.region_selector_gender_dropdown'),
		choices = [ 'none' ] + videokit.choices.region_selector_genders,
		value = state_manager.get_item('region_selector_gender') or 'none'
	)
	REGION_SELECTOR_RACE_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.region_selector_race_dropdown'),
		choices = [ 'none' ] + videokit.choices.region_selector_races,
		value = state_manager.get_item('region_selector_race') or 'none'
	)
	with gradio.Row():
		region_selector_age_start = state_manager.get_item('region_selector_age_start') or videokit.choices.region_selector_age_range[0]
		region_selector_age_end = state_manager.get_item('region_selector_age_end') or videokit.choices.region_selector_age_range[-1]
		REGION_SELECTOR_AGE_RANGE_SLIDER = RangeSlider(
			label = translator.get('uis.region_selector_age_range_slider'),
			minimum = videokit.choices.region_selector_age_range[0],
			maximum = videokit.choices.region_selector_age_range[-1],
			value = (region_selector_age_start, region_selector_age_end),
			step = calculate_int_step(videokit.choices.region_selector_age_range)
		)
	REFERENCE_REGION_DISTANCE_SLIDER = gradio.Slider(
		label = translator.get('uis.reference_region_distance_slider'),
		value = state_manager.get_item('reference_region_distance'),
		step = calculate_float_step(videokit.choices.reference_region_distance_range),
		minimum = videokit.choices.reference_region_distance_range[0],
		maximum = videokit.choices.reference_region_distance_range[-1],
		visible = 'reference' in state_manager.get_item('region_selector_mode')
	)
	register_ui_component('region_selector_mode_dropdown', REGION_SELECTOR_MODE_DROPDOWN)
	register_ui_component('region_selector_order_dropdown', REGION_SELECTOR_ORDER_DROPDOWN)
	register_ui_component('region_selector_gender_dropdown', REGION_SELECTOR_GENDER_DROPDOWN)
	register_ui_component('region_selector_race_dropdown', REGION_SELECTOR_RACE_DROPDOWN)
	register_ui_component('region_selector_age_range_slider', REGION_SELECTOR_AGE_RANGE_SLIDER)
	register_ui_component('reference_region_position_gallery', REFERENCE_REGION_POSITION_GALLERY)
	register_ui_component('reference_region_distance_slider', REFERENCE_REGION_DISTANCE_SLIDER)


def listen() -> None:
	region_selector_MODE_DROPDOWN.change(update_region_selector_mode, inputs = region_selector_MODE_DROPDOWN, outputs = [ REFERENCE_FACE_POSITION_GALLERY, REFERENCE_REGION_DISTANCE_SLIDER ])
	region_selector_ORDER_DROPDOWN.change(update_region_selector_order, inputs = region_selector_ORDER_DROPDOWN, outputs = REFERENCE_FACE_POSITION_GALLERY)
	region_selector_GENDER_DROPDOWN.change(update_region_selector_gender, inputs = region_selector_GENDER_DROPDOWN, outputs = REFERENCE_FACE_POSITION_GALLERY)
	region_selector_RACE_DROPDOWN.change(update_region_selector_race, inputs = region_selector_RACE_DROPDOWN, outputs = REFERENCE_FACE_POSITION_GALLERY)
	region_selector_AGE_RANGE_SLIDER.release(update_region_selector_age_range, inputs = region_selector_AGE_RANGE_SLIDER, outputs = REFERENCE_FACE_POSITION_GALLERY)
	REFERENCE_REGION_DISTANCE_SLIDER.release(update_reference_region_distance, inputs = REFERENCE_REGION_DISTANCE_SLIDER)

	preview_frame_slider = get_ui_component('preview_frame_slider')
	if preview_frame_slider:
		REFERENCE_FACE_POSITION_GALLERY.select(update_reference_frame_number, inputs = preview_frame_slider)
		REFERENCE_FACE_POSITION_GALLERY.select(update_reference_region_position)

	for ui_component in get_ui_components(
	[
		'target_image',
		'target_video'
	]):
		for method in [ 'change', 'clear' ]:
			getattr(ui_component, method)(clear_reference_frame_number)
			getattr(ui_component, method)(clear_reference_region_position)
			getattr(ui_component, method)(update_reference_position_gallery, outputs = REFERENCE_FACE_POSITION_GALLERY)

	for ui_component in get_ui_components(
	[
		'region_detector_model_dropdown',
		'region_detector_size_dropdown',
		'region_detector_angles_checkbox_group'
	]):
		ui_component.change(clear_and_update_reference_position_gallery, outputs = REFERENCE_REGION_POSITION_GALLERY)

	region_detector_score_slider = get_ui_component('region_detector_score_slider')
	if region_detector_score_slider:
		region_detector_score_slider.release(update_reference_position_gallery, outputs = REFERENCE_REGION_POSITION_GALLERY)

	preview_frame_slider = get_ui_component('preview_frame_slider')
	if preview_frame_slider:
		for method in [ 'change', 'release' ]:
			getattr(preview_frame_slider, method)(update_reference_position_gallery, inputs = preview_frame_slider, outputs = REFERENCE_REGION_POSITION_GALLERY, show_progress = 'hidden')


def update_region_selector_mode(region_selector_mode : RegionSelectorMode) -> Tuple[gradio.Gallery, gradio.Slider]:
	state_manager.set_item('region_selector_mode', region_selector_mode)
	if region_selector_mode == 'many':
		return gradio.Gallery(visible = False), gradio.Slider(visible = False)
	if region_selector_mode == 'one':
		return gradio.Gallery(visible = False), gradio.Slider(visible = False)
	if region_selector_mode == 'reference':
		return gradio.Gallery(visible = True), gradio.Slider(visible = True)


def update_region_selector_order(region_analyser_order : RegionSelectorOrder) -> gradio.Gallery:
	state_manager.set_item('region_selector_order', convert_str_none(region_analyser_order))
	return update_reference_position_gallery()


def update_region_selector_gender(region_selector_gender : Gender) -> gradio.Gallery:
	state_manager.set_item('region_selector_gender', convert_str_none(region_selector_gender))
	return update_reference_position_gallery()


def update_region_selector_race(region_selector_race : Race) -> gradio.Gallery:
	state_manager.set_item('region_selector_race', convert_str_none(region_selector_race))
	return update_reference_position_gallery()


def update_region_selector_age_range(region_selector_age_range : Tuple[float, float]) -> gradio.Gallery:
	region_selector_age_start, region_selector_age_end = region_selector_age_range
	state_manager.set_item('region_selector_age_start', int(region_selector_age_start))
	state_manager.set_item('region_selector_age_end', int(region_selector_age_end))
	return update_reference_position_gallery()


def update_reference_region_position(event : gradio.SelectData) -> None:
	state_manager.set_item('reference_region_position', event.index)


def clear_reference_region_position() -> None:
	state_manager.set_item('reference_region_position', 0)


def update_reference_region_distance(reference_region_distance : float) -> None:
	state_manager.set_item('reference_region_distance', reference_region_distance)


def update_reference_frame_number(reference_frame_number : int = 0) -> None:
	state_manager.set_item('reference_frame_number', reference_frame_number)


def clear_reference_frame_number() -> None:
	state_manager.set_item('reference_frame_number', 0)


def clear_and_update_reference_position_gallery() -> gradio.Gallery:
	clear_static_regions()
	return update_reference_position_gallery()


def update_reference_position_gallery(frame_number : int = 0) -> gradio.Gallery:
	gallery_vision_frames = []
	if is_image(state_manager.get_item('target_path')):
		target_vision_frame = read_static_image(state_manager.get_item('target_path'))
		gallery_vision_frames = extract_gallery_frames(target_vision_frame)
	if is_video(state_manager.get_item('target_path')):
		target_vision_frame = read_video_frame(state_manager.get_item('target_path'), frame_number)
		gallery_vision_frames = extract_gallery_frames(target_vision_frame)
	if gallery_vision_frames:
		return gradio.Gallery(value = gallery_vision_frames)
	return gradio.Gallery(value = None)


def extract_gallery_frames(target_vision_frame : VisionFrame) -> List[VisionFrame]:
	gallery_vision_frames = []
	regions = get_many_regions([ target_vision_frame ])
	regions = sort_and_filter_regions(regions)

	for region in regions:
		start_x, start_y, end_x, end_y = map(int, region.bounding_box)
		padding_x = int((end_x - start_x) * 0.25)
		padding_y = int((end_y - start_y) * 0.25)
		start_x = max(0, start_x - padding_x)
		start_y = max(0, start_y - padding_y)
		end_x = max(0, end_x + padding_x)
		end_y = max(0, end_y + padding_y)
		crop_vision_frame = target_vision_frame[start_y:end_y, start_x:end_x]
		crop_vision_frame = fit_cover_frame(crop_vision_frame, (128, 128))
		crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_BGR2RGB)
		gallery_vision_frames.append(crop_vision_frame)
	return gallery_vision_frames
