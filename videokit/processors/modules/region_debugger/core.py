from argparse import ArgumentParser

import cv2
import numpy

import videokit.jobs.job_manager
import videokit.jobs.job_store
from videokit import config, content_analyser, region_classifier, region_detector, landmark_detector, region_masker, region_recognizer, logger, state_manager, translator, video_manager
from videokit.region_analyser import scale_region
from videokit.region_helper import warp_region_by_region_landmark_5
from videokit.region_masker import create_area_mask, create_box_mask, create_occlusion_mask, create_region_mask
from videokit.region_selector import select_regions
from videokit.filesystem import in_directory, is_image, is_video, same_file_extension
from videokit.processors.modules.region_debugger import choices as region_debugger_choices
from videokit.processors.modules.region_debugger.types import RegionDebuggerInputs
from videokit.processors.types import ProcessorOutputs
from videokit.program_helper import find_argument_group
from videokit.types import ApplyStateItem, Args, Region, InferencePool, ProcessMode, VisionFrame
from videokit.vision import read_static_image, read_static_video_frame


def get_inference_pool() -> InferencePool:
	pass


def clear_inference_pool() -> None:
	pass


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--region-debugger-items', help = translator.get('help.items', __package__).format(choices = ', '.join(region_debugger_choices.region_debugger_items)), default = config.get_str_list('processors', 'region_debugger_items', 'region-landmark-5/68 region-mask'), choices = region_debugger_choices.region_debugger_items, nargs = '+', metavar = 'REGION_DEBUGGER_ITEMS')
		videokit.jobs.job_store.register_step_keys([ 'region_debugger_items' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('region_debugger_items', args.get('region_debugger_items'))


def pre_check() -> bool:
	return True


def pre_process(mode : ProcessMode) -> bool:
	return True


def post_process() -> None:
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()
	video_manager.clear_video_pool()
	if state_manager.get_item('video_memory_strategy') == 'strict':
		content_analyser.clear_inference_pool()
		region_classifier.clear_inference_pool()
		region_detector.clear_inference_pool()
		landmark_detector.clear_inference_pool()
		region_masker.clear_inference_pool()
		region_recognizer.clear_inference_pool()


def debug_region(target_region : Region, temp_vision_frame : VisionFrame) -> VisionFrame:
	region_debugger_items = state_manager.get_item('region_debugger_items')

	if 'bounding-box' in region_debugger_items:
		temp_vision_frame = draw_bounding_box(target_region, temp_vision_frame)

	if 'region-mask' in region_debugger_items:
		temp_vision_frame = draw_region_mask(target_region, temp_vision_frame)

	if 'region-landmark-5' in region_debugger_items:
		temp_vision_frame = draw_region_landmark_5(target_region, temp_vision_frame)

	if 'region-landmark-5/68' in region_debugger_items:
		temp_vision_frame = draw_region_landmark_5_68(target_region, temp_vision_frame)

	if 'region-landmark-68' in region_debugger_items:
		temp_vision_frame = draw_region_landmark_68(target_region, temp_vision_frame)

	if 'region-landmark-68/5' in region_debugger_items:
		temp_vision_frame = draw_region_landmark_68_5(target_region, temp_vision_frame)

	return temp_vision_frame


def draw_bounding_box(target_region : Region, temp_vision_frame : VisionFrame) -> VisionFrame:
	temp_vision_frame = numpy.ascontiguousarray(temp_vision_frame)
	box_color = 0, 0, 255
	border_color = 100, 100, 255
	bounding_box = target_region.bounding_box.astype(numpy.int32)
	x1, y1, x2, y2 = bounding_box

	cv2.rectangle(temp_vision_frame, (x1, y1), (x2, y2), box_color, 2)

	if target_region.angle == 0:
		cv2.line(temp_vision_frame, (x1, y1), (x2, y1), border_color, 3)
	if target_region.angle == 180:
		cv2.line(temp_vision_frame, (x1, y2), (x2, y2), border_color, 3)
	if target_region.angle == 90:
		cv2.line(temp_vision_frame, (x2, y1), (x2, y2), border_color, 3)
	if target_region.angle == 270:
		cv2.line(temp_vision_frame, (x1, y1), (x1, y2), border_color, 3)

	return temp_vision_frame


def draw_region_mask(target_region : Region, temp_vision_frame : VisionFrame) -> VisionFrame:
	crop_masks = []
	temp_vision_frame = numpy.ascontiguousarray(temp_vision_frame)
	region_landmark_5 = target_region.landmark_set.get('5')
	region_landmark_68 = target_region.landmark_set.get('68')
	region_landmark_5_68 = target_region.landmark_set.get('5/68')
	crop_vision_frame, affine_matrix = warp_region_by_region_landmark_5(temp_vision_frame, region_landmark_5_68, 'arcface_128', (512, 512))
	inverse_matrix = cv2.invertAffineTransform(affine_matrix)
	temp_size = temp_vision_frame.shape[:2][::-1]
	mask_color = 0, 255, 0

	if numpy.array_equal(region_landmark_5, region_landmark_5_68):
		mask_color = 255, 255, 0

	if 'box' in state_manager.get_item('region_mask_types'):
		box_mask = create_box_mask(crop_vision_frame, 0, state_manager.get_item('region_mask_padding'))
		crop_masks.append(box_mask)

	if 'occlusion' in state_manager.get_item('region_mask_types'):
		occlusion_mask = create_occlusion_mask(crop_vision_frame)
		crop_masks.append(occlusion_mask)

	if 'area' in state_manager.get_item('region_mask_types'):
		region_landmark_68 = cv2.transform(region_landmark_68.reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
		area_mask = create_area_mask(crop_vision_frame, region_landmark_68, state_manager.get_item('region_mask_areas'))
		crop_masks.append(area_mask)

	if 'region' in state_manager.get_item('region_mask_types'):
		region_mask = create_region_mask(crop_vision_frame, state_manager.get_item('region_mask_regions'))
		crop_masks.append(region_mask)

	crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
	crop_mask = (crop_mask * 255).astype(numpy.uint8)
	inverse_vision_frame = cv2.warpAffine(crop_mask, inverse_matrix, temp_size)
	inverse_vision_frame = cv2.threshold(inverse_vision_frame, 100, 255, cv2.THRESH_BINARY)[1]
	inverse_contours, _ = cv2.findContours(inverse_vision_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	cv2.drawContours(temp_vision_frame, inverse_contours, -1, mask_color, 2)

	return temp_vision_frame


def draw_region_landmark_5(target_region : Region, temp_vision_frame : VisionFrame) -> VisionFrame:
	temp_vision_frame = numpy.ascontiguousarray(temp_vision_frame)
	region_landmark_5 = target_region.landmark_set.get('5')
	point_color = 0, 0, 255

	if numpy.any(region_landmark_5):
		region_landmark_5 = region_landmark_5.astype(numpy.int32)

		for point in region_landmark_5:
			cv2.circle(temp_vision_frame, tuple(point), 3, point_color, -1)

	return temp_vision_frame


def draw_region_landmark_5_68(target_region : Region, temp_vision_frame : VisionFrame) -> VisionFrame:
	temp_vision_frame = numpy.ascontiguousarray(temp_vision_frame)
	region_landmark_5 = target_region.landmark_set.get('5')
	region_landmark_5_68 = target_region.landmark_set.get('5/68')
	point_color = 0, 255, 0

	if numpy.array_equal(region_landmark_5, region_landmark_5_68):
		point_color = 255, 255, 0

	if numpy.any(region_landmark_5_68):
		region_landmark_5_68 = region_landmark_5_68.astype(numpy.int32)

		for point in region_landmark_5_68:
			cv2.circle(temp_vision_frame, tuple(point), 3, point_color, -1)

	return temp_vision_frame


def draw_region_landmark_68(target_region : Region, temp_vision_frame : VisionFrame) -> VisionFrame:
	temp_vision_frame = numpy.ascontiguousarray(temp_vision_frame)
	region_landmark_68 = target_region.landmark_set.get('68')
	region_landmark_68_5 = target_region.landmark_set.get('68/5')
	point_color = 0, 255, 0

	if numpy.array_equal(region_landmark_68, region_landmark_68_5):
		point_color = 255, 255, 0

	if numpy.any(region_landmark_68):
		region_landmark_68 = region_landmark_68.astype(numpy.int32)

		for point in region_landmark_68:
			cv2.circle(temp_vision_frame, tuple(point), 3, point_color, -1)

	return temp_vision_frame


def draw_region_landmark_68_5(target_region : Region, temp_vision_frame : VisionFrame) -> VisionFrame:
	temp_vision_frame = numpy.ascontiguousarray(temp_vision_frame)
	region_landmark_68_5 = target_region.landmark_set.get('68/5')
	point_color = 255, 255, 0

	if numpy.any(region_landmark_68_5):
		region_landmark_68_5 = region_landmark_68_5.astype(numpy.int32)

		for point in region_landmark_68_5:
			cv2.circle(temp_vision_frame, tuple(point), 3, point_color, -1)

	return temp_vision_frame


def process_frame(inputs : RegionDebuggerInputs) -> ProcessorOutputs:
	reference_vision_frame = inputs.get('reference_vision_frame')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	target_regions = select_regions(reference_vision_frame, target_vision_frame)

	if target_regions:
		for target_region in target_regions:
			target_region = scale_region(target_region, target_vision_frame, temp_vision_frame)
			temp_vision_frame = debug_region(target_region, temp_vision_frame)

	return temp_vision_frame, temp_vision_mask


