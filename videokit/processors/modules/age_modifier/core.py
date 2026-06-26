from argparse import ArgumentParser
from functools import lru_cache

import cv2
import numpy

import videokit.choices
import videokit.jobs.job_manager
import videokit.jobs.job_store
from videokit import config, content_analyser, region_classifier, region_detector, landmark_detector, region_masker, region_recognizer, inference_manager, logger, state_manager, translator, video_manager
from videokit.common_helper import create_int_metavar, is_macos
from videokit.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from videokit.execution import has_execution_provider
from videokit.region_analyser import scale_region
from videokit.region_helper import merge_matrix, paste_back, scale_region_landmark_5, warp_region_by_region_landmark_5
from videokit.region_masker import create_box_mask, create_occlusion_mask
from videokit.region_selector import select_regions
from videokit.filesystem import in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from videokit.processors.modules.age_modifier import choices as age_modifier_choices
from videokit.processors.modules.age_modifier.types import AgeModifierDirection, AgeModifierInputs
from videokit.processors.types import ProcessorOutputs
from videokit.program_helper import find_argument_group
from videokit.thread_helper import thread_semaphore
from videokit.types import ApplyStateItem, Args, DownloadScope, Region, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from videokit.vision import match_frame_color, read_static_image, read_static_video_frame


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'fran':
		{
			'__metadata__':
			{
				'vendor': 'ry-lu',
				'license': 'mit',
				'year': 2024
			},
			'hashes':
			{
				'age_modifier':
				{
					'url': resolve_download_url('models-3.6.0', 'fran.hash'),
					'path': resolve_relative_path('../.assets/models/fran.hash')
				}
			},
			'sources':
			{
				'age_modifier':
				{
					'url': resolve_download_url('models-3.6.0', 'fran.onnx'),
					'path': resolve_relative_path('../.assets/models/fran.onnx')
				}
			},
			'templates':
			{
				'target': 'ffhq_512',
			},
			'sizes':
			{
				'target': (1024, 1024),
			},
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'styleganex_age':
		{
			'__metadata__':
			{
				'vendor': 'williamyang1991',
				'license': 'S-Lab-1.0',
				'year': 2023
			},
			'hashes':
			{
				'age_modifier':
				{
					'url': resolve_download_url('models-3.1.0', 'styleganex_age.hash'),
					'path': resolve_relative_path('../.assets/models/styleganex_age.hash')
				}
			},
			'sources':
			{
				'age_modifier':
				{
					'url': resolve_download_url('models-3.1.0', 'styleganex_age.onnx'),
					'path': resolve_relative_path('../.assets/models/styleganex_age.onnx')
				}
			},
			'templates':
			{
				'target': 'ffhq_512',
				'target_with_background': 'styleganex_384'
			},
			'sizes':
			{
				'target': (256, 256),
				'target_with_background': (384, 384)
			},
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ state_manager.get_item('age_modifier_model') ]
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ state_manager.get_item('age_modifier_model') ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
	model_name = state_manager.get_item('age_modifier_model')
	return create_static_model_set('full').get(model_name)


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--age-modifier-model', help = translator.get('help.model', __package__), default = config.get_str_value('processors', 'age_modifier_model', 'fran'), choices = age_modifier_choices.age_modifier_models)
		group_processors.add_argument('--age-modifier-direction', help = translator.get('help.direction', __package__), type = int, default = config.get_int_value('processors', 'age_modifier_direction', '0'), choices = age_modifier_choices.age_modifier_direction_range, metavar = create_int_metavar(age_modifier_choices.age_modifier_direction_range))
		videokit.jobs.job_store.register_step_keys([ 'age_modifier_model', 'age_modifier_direction' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('age_modifier_model', args.get('age_modifier_model'))
	apply_state_item('age_modifier_direction', args.get('age_modifier_direction'))


def pre_check() -> bool:
	return True


def pre_process(mode : ProcessMode) -> bool:
	return True


def post_process() -> None:
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()
	video_manager.clear_video_pool()
	if state_manager.get_item('video_memory_strategy') in [ 'strict', 'moderate' ]:
		clear_inference_pool()
	if state_manager.get_item('video_memory_strategy') == 'strict':
		content_analyser.clear_inference_pool()
		region_classifier.clear_inference_pool()
		region_detector.clear_inference_pool()
		landmark_detector.clear_inference_pool()
		region_masker.clear_inference_pool()
		region_recognizer.clear_inference_pool()


def modify_age(target_region : Region, temp_vision_frame : VisionFrame) -> VisionFrame:
	model_templates = get_model_options().get('templates')
	model_sizes = get_model_options().get('sizes')
	region_landmark_5 = target_region.landmark_set.get('5/68').copy()
	crop_vision_frame, affine_matrix = warp_region_by_region_landmark_5(temp_vision_frame, region_landmark_5, model_templates.get('target'), model_sizes.get('target'))

	if state_manager.get_item('age_modifier_model') == 'fran':
		box_mask = create_box_mask(crop_vision_frame, state_manager.get_item('region_mask_blur'), (0, 0, 0, 0))
		crop_masks =\
		[
			box_mask
		]

		if 'occlusion' in state_manager.get_item('region_mask_types'):
			occlusion_mask = create_occlusion_mask(crop_vision_frame)
			crop_masks.append(occlusion_mask)

		crop_vision_frame = prepare_vision_frame(crop_vision_frame)
		target_age = numpy.mean(target_region.age)
		age_modifier_direction = numpy.array([ target_age, target_age + state_manager.get_item('age_modifier_direction') ], dtype = numpy.float32) / 100
		age_modifier_direction = age_modifier_direction.clip(0, 1)
		crop_vision_frame = forward(crop_vision_frame, crop_vision_frame, age_modifier_direction)
		crop_vision_frame = normalize_vision_frame(crop_vision_frame)
		crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
		paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
		return paste_vision_frame

	if state_manager.get_item('age_modifier_model') == 'styleganex_age':
		extend_region_landmark_5 = scale_region_landmark_5(region_landmark_5, 0.875)
		extend_vision_frame, extend_affine_matrix = warp_region_by_region_landmark_5(temp_vision_frame, extend_region_landmark_5, model_templates.get('target_with_background'), model_sizes.get('target_with_background'))
		extend_vision_frame_raw = extend_vision_frame.copy()
		box_mask = create_box_mask(extend_vision_frame, state_manager.get_item('region_mask_blur'), (0, 0, 0, 0))
		crop_masks =\
		[
			box_mask
		]

		if 'occlusion' in state_manager.get_item('region_mask_types'):
			occlusion_mask = create_occlusion_mask(crop_vision_frame)
			temp_matrix = merge_matrix([ extend_affine_matrix, cv2.invertAffineTransform(affine_matrix) ])
			occlusion_mask = cv2.warpAffine(occlusion_mask, temp_matrix, model_sizes.get('target_with_background'))
			crop_masks.append(occlusion_mask)

		crop_vision_frame = prepare_vision_frame(crop_vision_frame)
		extend_vision_frame = prepare_vision_frame(extend_vision_frame)
		age_modifier_direction = numpy.array(numpy.interp(state_manager.get_item('age_modifier_direction'), [ -100, 100 ], [ 2.5, -2.5 ])).astype(numpy.float32)
		extend_vision_frame = forward(crop_vision_frame, extend_vision_frame, age_modifier_direction)
		extend_vision_frame = normalize_extend_frame(extend_vision_frame)
		extend_vision_frame = match_frame_color(extend_vision_frame_raw, extend_vision_frame)
		extend_affine_matrix *= (model_sizes.get('target')[0] * 4) / model_sizes.get('target_with_background')[0]
		crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
		crop_mask = cv2.resize(crop_mask, (model_sizes.get('target')[0] * 4, model_sizes.get('target')[1] * 4))
		paste_vision_frame = paste_back(temp_vision_frame, extend_vision_frame, crop_mask, extend_affine_matrix)
		return paste_vision_frame

	return temp_vision_frame


def forward(crop_vision_frame : VisionFrame, extend_vision_frame : VisionFrame, age_modifier_direction : AgeModifierDirection) -> VisionFrame:
	age_modifier = get_inference_pool().get('age_modifier')
	age_modifier_inputs = {}

	if is_macos() and has_execution_provider('coreml'):
		age_modifier.set_providers([ videokit.choices.execution_provider_set.get('cpu') ])

	for age_modifier_input in age_modifier.get_inputs():
		if age_modifier_input.name == 'target':
			age_modifier_inputs[age_modifier_input.name] = crop_vision_frame
		if age_modifier_input.name == 'target_with_background':
			age_modifier_inputs[age_modifier_input.name] = extend_vision_frame
		if age_modifier_input.name == 'direction':
			age_modifier_inputs[age_modifier_input.name] = age_modifier_direction

	with thread_semaphore():
		crop_vision_frame = age_modifier.run(None, age_modifier_inputs)[0][0]

	return crop_vision_frame


def prepare_vision_frame(vision_frame : VisionFrame) -> VisionFrame:
	model_mean = get_model_options().get('mean')
	model_standard_deviation = get_model_options().get('standard_deviation')
	vision_frame = vision_frame[:, :, ::-1] / 255.0
	vision_frame = (vision_frame - model_mean) / model_standard_deviation
	vision_frame = numpy.expand_dims(vision_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
	return vision_frame


def normalize_vision_frame(vision_frame : VisionFrame) -> VisionFrame:
	model_mean = get_model_options().get('mean')
	model_standard_deviation = get_model_options().get('standard_deviation')
	vision_frame = vision_frame.transpose(1, 2, 0)
	vision_frame = vision_frame * model_standard_deviation + model_mean
	vision_frame = vision_frame.clip(0, 1)
	vision_frame = vision_frame[:, :, ::-1] * 255
	return vision_frame


def normalize_extend_frame(extend_vision_frame : VisionFrame) -> VisionFrame:
	model_sizes = get_model_options().get('sizes')
	extend_vision_frame = numpy.clip(extend_vision_frame, -1, 1)
	extend_vision_frame = (extend_vision_frame + 1) / 2
	extend_vision_frame = extend_vision_frame.transpose(1, 2, 0).clip(0, 255)
	extend_vision_frame = (extend_vision_frame * 255.0)
	extend_vision_frame = extend_vision_frame.astype(numpy.uint8)[:, :, ::-1]
	extend_vision_frame = cv2.resize(extend_vision_frame, (model_sizes.get('target')[0] * 4, model_sizes.get('target')[1] * 4), interpolation = cv2.INTER_AREA)
	return extend_vision_frame


def process_frame(inputs : AgeModifierInputs) -> ProcessorOutputs:
	reference_vision_frame = inputs.get('reference_vision_frame')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	target_regions = select_regions(reference_vision_frame, target_vision_frame)

	if target_regions:
		for target_region in target_regions:
			target_region = scale_region(target_region, target_vision_frame, temp_vision_frame)
			temp_vision_frame = modify_age(target_region, temp_vision_frame)

	return temp_vision_frame, temp_vision_mask
