from argparse import ArgumentParser
from functools import lru_cache
from typing import Tuple

import cv2
import numpy

import videokit.jobs.job_manager
import videokit.jobs.job_store
from videokit import config, content_analyser, region_classifier, region_detector, landmark_detector, region_masker, region_recognizer, inference_manager, logger, state_manager, translator, video_manager
from videokit.common_helper import create_float_metavar
from videokit.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from videokit.region_analyser import scale_region
from videokit.region_helper import paste_back, scale_region_landmark_5, warp_region_by_region_landmark_5
from videokit.region_masker import create_box_mask
from videokit.region_selector import select_regions
from videokit.filesystem import in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from videokit.processors.live_portrait import create_rotation, limit_angle, limit_expression
from videokit.processors.modules.portrait_editor import choices as portrait_editor_choices
from videokit.processors.modules.portrait_editor.types import PortraitEditorInputs
from videokit.processors.types import LivePortraitExpression, LivePortraitFeatureVolume, LivePortraitMotionPoints, LivePortraitPitch, LivePortraitRoll, LivePortraitRotation, LivePortraitScale, LivePortraitTranslation, LivePortraitYaw, ProcessorOutputs
from videokit.program_helper import find_argument_group
from videokit.thread_helper import conditional_thread_semaphore, thread_semaphore
from videokit.types import ApplyStateItem, Args, DownloadScope, Region, RegionLandmark68, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from videokit.vision import read_static_image, read_static_video_frame


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'live_portrait':
		{
			'__metadata__':
			{
				'vendor': 'KwaiVGI',
				'license': 'MIT',
				'year': 2024
			},
			'hashes':
			{
				'feature_extractor':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_feature_extractor.hash'),
					'path': resolve_relative_path('../.assets/models/live_portrait_feature_extractor.hash')
				},
				'motion_extractor':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_motion_extractor.hash'),
					'path': resolve_relative_path('../.assets/models/live_portrait_motion_extractor.hash')
				},
				'eye_retargeter':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_eye_retargeter.hash'),
					'path': resolve_relative_path('../.assets/models/live_portrait_eye_retargeter.hash')
				},
				'lip_retargeter':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_lip_retargeter.hash'),
					'path': resolve_relative_path('../.assets/models/live_portrait_lip_retargeter.hash')
				},
				'stitcher':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_stitcher.hash'),
					'path': resolve_relative_path('../.assets/models/live_portrait_stitcher.hash')
				},
				'generator':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_generator.hash'),
					'path': resolve_relative_path('../.assets/models/live_portrait_generator.hash')
				}
			},
			'sources':
			{
				'feature_extractor':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_feature_extractor.onnx'),
					'path': resolve_relative_path('../.assets/models/live_portrait_feature_extractor.onnx')
				},
				'motion_extractor':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_motion_extractor.onnx'),
					'path': resolve_relative_path('../.assets/models/live_portrait_motion_extractor.onnx')
				},
				'eye_retargeter':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_eye_retargeter.onnx'),
					'path': resolve_relative_path('../.assets/models/live_portrait_eye_retargeter.onnx')
				},
				'lip_retargeter':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_lip_retargeter.onnx'),
					'path': resolve_relative_path('../.assets/models/live_portrait_lip_retargeter.onnx')
				},
				'stitcher':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_stitcher.onnx'),
					'path': resolve_relative_path('../.assets/models/live_portrait_stitcher.onnx')
				},
				'generator':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_generator.onnx'),
					'path': resolve_relative_path('../.assets/models/live_portrait_generator.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ state_manager.get_item('portrait_editor_model') ]
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ state_manager.get_item('portrait_editor_model') ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
	model_name = state_manager.get_item('portrait_editor_model')
	return create_static_model_set('full').get(model_name)


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--portrait-editor-model', help = translator.get('help.model', __package__), default = config.get_str_value('processors', 'portrait_editor_model', 'live_portrait'), choices = portrait_editor_choices.portrait_editor_models)
		group_processors.add_argument('--portrait-editor-eyebrow-direction', help = translator.get('help.eyebrow_direction', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_eyebrow_direction', '0'), choices = portrait_editor_choices.portrait_editor_eyebrow_direction_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_eyebrow_direction_range))
		group_processors.add_argument('--portrait-editor-eye-gaze-horizontal', help = translator.get('help.eye_gaze_horizontal', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_eye_gaze_horizontal', '0'), choices = portrait_editor_choices.portrait_editor_eye_gaze_horizontal_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_eye_gaze_horizontal_range))
		group_processors.add_argument('--portrait-editor-eye-gaze-vertical', help = translator.get('help.eye_gaze_vertical', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_eye_gaze_vertical', '0'), choices = portrait_editor_choices.portrait_editor_eye_gaze_vertical_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_eye_gaze_vertical_range))
		group_processors.add_argument('--portrait-editor-eye-open-ratio', help = translator.get('help.eye_open_ratio', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_eye_open_ratio', '0'), choices = portrait_editor_choices.portrait_editor_eye_open_ratio_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_eye_open_ratio_range))
		group_processors.add_argument('--portrait-editor-lip-open-ratio', help = translator.get('help.lip_open_ratio', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_lip_open_ratio', '0'), choices = portrait_editor_choices.portrait_editor_lip_open_ratio_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_lip_open_ratio_range))
		group_processors.add_argument('--portrait-editor-mouth-grim', help = translator.get('help.mouth_grim', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_mouth_grim', '0'), choices = portrait_editor_choices.portrait_editor_mouth_grim_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_mouth_grim_range))
		group_processors.add_argument('--portrait-editor-mouth-pout', help = translator.get('help.mouth_pout', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_mouth_pout', '0'), choices = portrait_editor_choices.portrait_editor_mouth_pout_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_mouth_pout_range))
		group_processors.add_argument('--portrait-editor-mouth-purse', help = translator.get('help.mouth_purse', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_mouth_purse', '0'), choices = portrait_editor_choices.portrait_editor_mouth_purse_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_mouth_purse_range))
		group_processors.add_argument('--portrait-editor-mouth-smile', help = translator.get('help.mouth_smile', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_mouth_smile', '0'), choices = portrait_editor_choices.portrait_editor_mouth_smile_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_mouth_smile_range))
		group_processors.add_argument('--portrait-editor-mouth-position-horizontal', help = translator.get('help.mouth_position_horizontal', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_mouth_position_horizontal', '0'), choices = portrait_editor_choices.portrait_editor_mouth_position_horizontal_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_mouth_position_horizontal_range))
		group_processors.add_argument('--portrait-editor-mouth-position-vertical', help = translator.get('help.mouth_position_vertical', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_mouth_position_vertical', '0'), choices = portrait_editor_choices.portrait_editor_mouth_position_vertical_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_mouth_position_vertical_range))
		group_processors.add_argument('--portrait-editor-head-pitch', help = translator.get('help.head_pitch', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_head_pitch', '0'), choices = portrait_editor_choices.portrait_editor_head_pitch_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_head_pitch_range))
		group_processors.add_argument('--portrait-editor-head-yaw', help = translator.get('help.head_yaw', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_head_yaw', '0'), choices = portrait_editor_choices.portrait_editor_head_yaw_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_head_yaw_range))
		group_processors.add_argument('--portrait-editor-head-roll', help = translator.get('help.head_roll', __package__), type = float, default = config.get_float_value('processors', 'portrait_editor_head_roll', '0'), choices = portrait_editor_choices.portrait_editor_head_roll_range, metavar = create_float_metavar(portrait_editor_choices.portrait_editor_head_roll_range))
		videokit.jobs.job_store.register_step_keys([ 'portrait_editor_model', 'portrait_editor_eyebrow_direction', 'portrait_editor_eye_gaze_horizontal', 'portrait_editor_eye_gaze_vertical', 'portrait_editor_eye_open_ratio', 'portrait_editor_lip_open_ratio', 'portrait_editor_mouth_grim', 'portrait_editor_mouth_pout', 'portrait_editor_mouth_purse', 'portrait_editor_mouth_smile', 'portrait_editor_mouth_position_horizontal', 'portrait_editor_mouth_position_vertical', 'portrait_editor_head_pitch', 'portrait_editor_head_yaw', 'portrait_editor_head_roll' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('portrait_editor_model', args.get('portrait_editor_model'))
	apply_state_item('portrait_editor_eyebrow_direction', args.get('portrait_editor_eyebrow_direction'))
	apply_state_item('portrait_editor_eye_gaze_horizontal', args.get('portrait_editor_eye_gaze_horizontal'))
	apply_state_item('portrait_editor_eye_gaze_vertical', args.get('portrait_editor_eye_gaze_vertical'))
	apply_state_item('portrait_editor_eye_open_ratio', args.get('portrait_editor_eye_open_ratio'))
	apply_state_item('portrait_editor_lip_open_ratio', args.get('portrait_editor_lip_open_ratio'))
	apply_state_item('portrait_editor_mouth_grim', args.get('portrait_editor_mouth_grim'))
	apply_state_item('portrait_editor_mouth_pout', args.get('portrait_editor_mouth_pout'))
	apply_state_item('portrait_editor_mouth_purse', args.get('portrait_editor_mouth_purse'))
	apply_state_item('portrait_editor_mouth_smile', args.get('portrait_editor_mouth_smile'))
	apply_state_item('portrait_editor_mouth_position_horizontal', args.get('portrait_editor_mouth_position_horizontal'))
	apply_state_item('portrait_editor_mouth_position_vertical', args.get('portrait_editor_mouth_position_vertical'))
	apply_state_item('portrait_editor_head_pitch', args.get('portrait_editor_head_pitch'))
	apply_state_item('portrait_editor_head_yaw', args.get('portrait_editor_head_yaw'))
	apply_state_item('portrait_editor_head_roll', args.get('portrait_editor_head_roll'))


def pre_check() -> bool:
	model_hash_set = get_model_options().get('hashes')
	model_source_set = get_model_options().get('sources')

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def pre_process(mode : ProcessMode) -> bool:
	if mode in [ 'output', 'preview' ] and not is_image(state_manager.get_item('target_path')) and not is_video(state_manager.get_item('target_path')):
		logger.error(translator.get('choose_image_or_video_target') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
		logger.error(translator.get('specify_image_or_video_output') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not same_file_extension(state_manager.get_item('target_path'), state_manager.get_item('output_path')):
		logger.error(translator.get('match_target_and_output_extension') + translator.get('exclamation_mark'), __name__)
		return False
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


def edit_portrait(target_region : Region, temp_vision_frame : VisionFrame) -> VisionFrame:
	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	region_landmark_5 = scale_region_landmark_5(target_region.landmark_set.get('5/68'), 1.5)
	crop_vision_frame, affine_matrix = warp_region_by_region_landmark_5(temp_vision_frame, region_landmark_5, model_template, model_size)
	box_mask = create_box_mask(crop_vision_frame, state_manager.get_item('region_mask_blur'), (0, 0, 0, 0))
	crop_vision_frame = prepare_crop_frame(crop_vision_frame)
	crop_vision_frame = apply_edit(crop_vision_frame, target_region.landmark_set.get('68'))
	crop_vision_frame = normalize_crop_frame(crop_vision_frame)
	paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, box_mask, affine_matrix)
	return paste_vision_frame


def apply_edit(crop_vision_frame : VisionFrame, region_landmark_68 : RegionLandmark68) -> VisionFrame:
	feature_volume = forward_extract_feature(crop_vision_frame)
	pitch, yaw, roll, scale, translation, expression, motion_points = forward_extract_motion(crop_vision_frame)
	rotation = create_rotation(pitch, yaw, roll)
	motion_points_target = scale * (motion_points @ rotation.T + expression) + translation
	expression = edit_eye_gaze(expression)
	expression = edit_mouth_grim(expression)
	expression = edit_mouth_position(expression)
	expression = edit_mouth_pout(expression)
	expression = edit_mouth_purse(expression)
	expression = edit_mouth_smile(expression)
	expression = edit_eyebrow_direction(expression)
	expression = limit_expression(expression)
	rotation = edit_head_rotation(pitch, yaw, roll)
	motion_points_source = motion_points @ rotation.T
	motion_points_source += expression
	motion_points_source *= scale
	motion_points_source += translation
	motion_points_source += edit_eye_open(motion_points_target, region_landmark_68)
	motion_points_source += edit_lip_open(motion_points_target, region_landmark_68)
	motion_points_source = forward_stitch_motion_points(motion_points_source, motion_points_target)
	crop_vision_frame = forward_generate_frame(feature_volume, motion_points_source, motion_points_target)
	return crop_vision_frame


def forward_extract_feature(crop_vision_frame : VisionFrame) -> LivePortraitFeatureVolume:
	feature_extractor = get_inference_pool().get('feature_extractor')

	with conditional_thread_semaphore():
		feature_volume = feature_extractor.run(None,
		{
			'input': crop_vision_frame
		})[0]

	return feature_volume


def forward_extract_motion(crop_vision_frame : VisionFrame) -> Tuple[LivePortraitPitch, LivePortraitYaw, LivePortraitRoll, LivePortraitScale, LivePortraitTranslation, LivePortraitExpression, LivePortraitMotionPoints]:
	motion_extractor = get_inference_pool().get('motion_extractor')

	with conditional_thread_semaphore():
		pitch, yaw, roll, scale, translation, expression, motion_points = motion_extractor.run(None,
		{
			'input': crop_vision_frame
		})

	return pitch, yaw, roll, scale, translation, expression, motion_points


def forward_retarget_eye(eye_motion_points : LivePortraitMotionPoints) -> LivePortraitMotionPoints:
	eye_retargeter = get_inference_pool().get('eye_retargeter')

	with conditional_thread_semaphore():
		eye_motion_points = eye_retargeter.run(None,
		{
			'input': eye_motion_points
		})[0]

	return eye_motion_points


def forward_retarget_lip(lip_motion_points : LivePortraitMotionPoints) -> LivePortraitMotionPoints:
	lip_retargeter = get_inference_pool().get('lip_retargeter')

	with conditional_thread_semaphore():
		lip_motion_points = lip_retargeter.run(None,
		{
			'input': lip_motion_points
		})[0]

	return lip_motion_points


def forward_stitch_motion_points(source_motion_points : LivePortraitMotionPoints, target_motion_points : LivePortraitMotionPoints) -> LivePortraitMotionPoints:
	stitcher = get_inference_pool().get('stitcher')

	with thread_semaphore():
		motion_points = stitcher.run(None,
		{
			'source': source_motion_points,
			'target': target_motion_points
		})[0]

	return motion_points


def forward_generate_frame(feature_volume : LivePortraitFeatureVolume, source_motion_points : LivePortraitMotionPoints, target_motion_points : LivePortraitMotionPoints) -> VisionFrame:
	generator = get_inference_pool().get('generator')

	with thread_semaphore():
		crop_vision_frame = generator.run(None,
		{
			'feature_volume': feature_volume,
			'source': source_motion_points,
			'target': target_motion_points
		})[0][0]

	return crop_vision_frame


def edit_eyebrow_direction(expression : LivePortraitExpression) -> LivePortraitExpression:
	portrait_editor_eyebrow = state_manager.get_item('portrait_editor_eyebrow_direction')

	if portrait_editor_eyebrow > 0:
		expression[0, 1, 1] += numpy.interp(portrait_editor_eyebrow, [ -1, 1 ], [ -0.015, 0.015 ])
		expression[0, 2, 1] -= numpy.interp(portrait_editor_eyebrow, [ -1, 1 ], [ -0.020, 0.020 ])
	else:
		expression[0, 1, 0] -= numpy.interp(portrait_editor_eyebrow, [ -1, 1 ], [ -0.015, 0.015 ])
		expression[0, 2, 0] += numpy.interp(portrait_editor_eyebrow, [ -1, 1 ], [ -0.020, 0.020 ])
		expression[0, 1, 1] += numpy.interp(portrait_editor_eyebrow, [ -1, 1 ], [ -0.005, 0.005 ])
		expression[0, 2, 1] -= numpy.interp(portrait_editor_eyebrow, [ -1, 1 ], [ -0.005, 0.005 ])
	return expression


def edit_eye_gaze(expression : LivePortraitExpression) -> LivePortraitExpression:
	portrait_editor_eye_gaze_horizontal = state_manager.get_item('portrait_editor_eye_gaze_horizontal')
	portrait_editor_eye_gaze_vertical = state_manager.get_item('portrait_editor_eye_gaze_vertical')

	if portrait_editor_eye_gaze_horizontal > 0:
		expression[0, 11, 0] += numpy.interp(portrait_editor_eye_gaze_horizontal, [ -1, 1 ], [ -0.015, 0.015 ])
		expression[0, 15, 0] += numpy.interp(portrait_editor_eye_gaze_horizontal, [ -1, 1 ], [ -0.020, 0.020 ])
	else:
		expression[0, 11, 0] += numpy.interp(portrait_editor_eye_gaze_horizontal, [ -1, 1 ], [ -0.020, 0.020 ])
		expression[0, 15, 0] += numpy.interp(portrait_editor_eye_gaze_horizontal, [ -1, 1 ], [ -0.015, 0.015 ])
	expression[0, 1, 1] += numpy.interp(portrait_editor_eye_gaze_vertical, [ -1, 1 ], [ -0.0025, 0.0025 ])
	expression[0, 2, 1] -= numpy.interp(portrait_editor_eye_gaze_vertical, [ -1, 1 ], [ -0.0025, 0.0025 ])
	expression[0, 11, 1] -= numpy.interp(portrait_editor_eye_gaze_vertical, [ -1, 1 ], [ -0.010, 0.010 ])
	expression[0, 13, 1] -= numpy.interp(portrait_editor_eye_gaze_vertical, [ -1, 1 ], [ -0.005, 0.005 ])
	expression[0, 15, 1] -= numpy.interp(portrait_editor_eye_gaze_vertical, [ -1, 1 ], [ -0.010, 0.010 ])
	expression[0, 16, 1] -= numpy.interp(portrait_editor_eye_gaze_vertical, [ -1, 1 ], [ -0.005, 0.005 ])
	return expression


def edit_eye_open(motion_points : LivePortraitMotionPoints, region_landmark_68 : RegionLandmark68) -> LivePortraitMotionPoints:
	portrait_editor_eye_open_ratio = state_manager.get_item('portrait_editor_eye_open_ratio')
	left_eye_ratio = calculate_distance_ratio(region_landmark_68, 37, 40, 39, 36)
	right_eye_ratio = calculate_distance_ratio(region_landmark_68, 43, 46, 45, 42)

	if portrait_editor_eye_open_ratio < 0:
		eye_motion_points = numpy.concatenate([ motion_points.ravel(), [ left_eye_ratio, right_eye_ratio, 0.0 ] ])
	else:
		eye_motion_points = numpy.concatenate([ motion_points.ravel(), [ left_eye_ratio, right_eye_ratio, 0.6 ] ])
	eye_motion_points = eye_motion_points.reshape(1, -1).astype(numpy.float32)
	eye_motion_points = forward_retarget_eye(eye_motion_points) * numpy.abs(portrait_editor_eye_open_ratio)
	eye_motion_points = eye_motion_points.reshape(-1, 21, 3)
	return eye_motion_points


def edit_lip_open(motion_points : LivePortraitMotionPoints, region_landmark_68 : RegionLandmark68) -> LivePortraitMotionPoints:
	portrait_editor_lip_open_ratio = state_manager.get_item('portrait_editor_lip_open_ratio')
	lip_ratio = calculate_distance_ratio(region_landmark_68, 62, 66, 54, 48)

	if portrait_editor_lip_open_ratio < 0:
		lip_motion_points = numpy.concatenate([ motion_points.ravel(), [ lip_ratio, 0.0 ] ])
	else:
		lip_motion_points = numpy.concatenate([ motion_points.ravel(), [ lip_ratio, 1.0 ] ])
	lip_motion_points = lip_motion_points.reshape(1, -1).astype(numpy.float32)
	lip_motion_points = forward_retarget_lip(lip_motion_points) * numpy.abs(portrait_editor_lip_open_ratio)
	lip_motion_points = lip_motion_points.reshape(-1, 21, 3)
	return lip_motion_points


def edit_mouth_grim(expression : LivePortraitExpression) -> LivePortraitExpression:
	portrait_editor_mouth_grim = state_manager.get_item('portrait_editor_mouth_grim')
	if portrait_editor_mouth_grim > 0:
		expression[0, 17, 2] -= numpy.interp(portrait_editor_mouth_grim, [ -1, 1 ], [ -0.005, 0.005 ])
		expression[0, 19, 2] += numpy.interp(portrait_editor_mouth_grim, [ -1, 1 ], [ -0.01, 0.01 ])
		expression[0, 20, 1] -= numpy.interp(portrait_editor_mouth_grim, [ -1, 1 ], [ -0.06, 0.06 ])
		expression[0, 20, 2] -= numpy.interp(portrait_editor_mouth_grim, [ -1, 1 ], [ -0.03, 0.03 ])
	else:
		expression[0, 19, 1] -= numpy.interp(portrait_editor_mouth_grim, [ -1, 1 ], [ -0.05, 0.05 ])
		expression[0, 19, 2] -= numpy.interp(portrait_editor_mouth_grim, [ -1, 1 ], [ -0.02, 0.02 ])
		expression[0, 20, 2] -= numpy.interp(portrait_editor_mouth_grim, [ -1, 1 ], [ -0.03, 0.03 ])
	return expression


def edit_mouth_position(expression : LivePortraitExpression) -> LivePortraitExpression:
	portrait_editor_mouth_position_horizontal = state_manager.get_item('portrait_editor_mouth_position_horizontal')
	portrait_editor_mouth_position_vertical = state_manager.get_item('portrait_editor_mouth_position_vertical')
	expression[0, 19, 0] += numpy.interp(portrait_editor_mouth_position_horizontal, [ -1, 1 ], [ -0.05, 0.05 ])
	expression[0, 20, 0] += numpy.interp(portrait_editor_mouth_position_horizontal, [ -1, 1 ], [ -0.04, 0.04 ])
	if portrait_editor_mouth_position_vertical > 0:
		expression[0, 19, 1] -= numpy.interp(portrait_editor_mouth_position_vertical, [ -1, 1 ], [ -0.04, 0.04 ])
		expression[0, 20, 1] -= numpy.interp(portrait_editor_mouth_position_vertical, [ -1, 1 ], [ -0.02, 0.02 ])
	else:
		expression[0, 19, 1] -= numpy.interp(portrait_editor_mouth_position_vertical, [ -1, 1 ], [ -0.05, 0.05 ])
		expression[0, 20, 1] -= numpy.interp(portrait_editor_mouth_position_vertical, [ -1, 1 ], [ -0.04, 0.04 ])
	return expression


def edit_mouth_pout(expression : LivePortraitExpression) -> LivePortraitExpression:
	portrait_editor_mouth_pout = state_manager.get_item('portrait_editor_mouth_pout')
	if portrait_editor_mouth_pout > 0:
		expression[0, 19, 1] -= numpy.interp(portrait_editor_mouth_pout, [ -1, 1 ], [ -0.022, 0.022 ])
		expression[0, 19, 2] += numpy.interp(portrait_editor_mouth_pout, [ -1, 1 ], [ -0.025, 0.025 ])
		expression[0, 20, 2] -= numpy.interp(portrait_editor_mouth_pout, [ -1, 1 ], [ -0.002, 0.002 ])
	else:
		expression[0, 19, 1] += numpy.interp(portrait_editor_mouth_pout, [ -1, 1 ], [ -0.022, 0.022 ])
		expression[0, 19, 2] += numpy.interp(portrait_editor_mouth_pout, [ -1, 1 ], [ -0.025, 0.025 ])
		expression[0, 20, 2] -= numpy.interp(portrait_editor_mouth_pout, [ -1, 1 ], [ -0.002, 0.002 ])
	return expression


def edit_mouth_purse(expression : LivePortraitExpression) -> LivePortraitExpression:
	portrait_editor_mouth_purse = state_manager.get_item('portrait_editor_mouth_purse')
	if portrait_editor_mouth_purse > 0:
		expression[0, 19, 1] -= numpy.interp(portrait_editor_mouth_purse, [ -1, 1 ], [ -0.04, 0.04 ])
		expression[0, 19, 2] -= numpy.interp(portrait_editor_mouth_purse, [ -1, 1 ], [ -0.02, 0.02 ])
	else:
		expression[0, 14, 1] -= numpy.interp(portrait_editor_mouth_purse, [ -1, 1 ], [ -0.02, 0.02 ])
		expression[0, 17, 2] += numpy.interp(portrait_editor_mouth_purse, [ -1, 1 ], [ -0.01, 0.01 ])
		expression[0, 19, 2] -= numpy.interp(portrait_editor_mouth_purse, [ -1, 1 ], [ -0.015, 0.015 ])
		expression[0, 20, 2] -= numpy.interp(portrait_editor_mouth_purse, [ -1, 1 ], [ -0.002, 0.002 ])
	return expression


def edit_mouth_smile(expression : LivePortraitExpression) -> LivePortraitExpression:
	portrait_editor_mouth_smile = state_manager.get_item('portrait_editor_mouth_smile')
	if portrait_editor_mouth_smile > 0:
		expression[0, 20, 1] -= numpy.interp(portrait_editor_mouth_smile, [ -1, 1 ], [ -0.015, 0.015 ])
		expression[0, 14, 1] -= numpy.interp(portrait_editor_mouth_smile, [ -1, 1 ], [ -0.025, 0.025 ])
		expression[0, 17, 1] += numpy.interp(portrait_editor_mouth_smile, [ -1, 1 ], [ -0.01, 0.01 ])
		expression[0, 17, 2] += numpy.interp(portrait_editor_mouth_smile, [ -1, 1 ], [ -0.004, 0.004 ])
		expression[0, 3, 1] -= numpy.interp(portrait_editor_mouth_smile, [ -1, 1 ], [ -0.0045, 0.0045 ])
		expression[0, 7, 1] -= numpy.interp(portrait_editor_mouth_smile, [ -1, 1 ], [ -0.0045, 0.0045 ])
	else:
		expression[0, 14, 1] -= numpy.interp(portrait_editor_mouth_smile, [ -1, 1 ], [ -0.02, 0.02 ])
		expression[0, 17, 1] += numpy.interp(portrait_editor_mouth_smile, [ -1, 1 ], [ -0.003, 0.003 ])
		expression[0, 19, 1] += numpy.interp(portrait_editor_mouth_smile, [ -1, 1 ], [ -0.02, 0.02 ])
		expression[0, 19, 2] -= numpy.interp(portrait_editor_mouth_smile, [ -1, 1 ], [ -0.005, 0.005 ])
		expression[0, 20, 2] += numpy.interp(portrait_editor_mouth_smile, [ -1, 1 ], [ -0.01, 0.01 ])
		expression[0, 3, 1] += numpy.interp(portrait_editor_mouth_smile, [ -1, 1 ], [ -0.0045, 0.0045 ])
		expression[0, 7, 1] += numpy.interp(portrait_editor_mouth_smile, [ -1, 1 ], [ -0.0045, 0.0045 ])
	return expression


def edit_head_rotation(pitch : LivePortraitPitch, yaw : LivePortraitYaw, roll : LivePortraitRoll) -> LivePortraitRotation:
	portrait_editor_head_pitch = state_manager.get_item('portrait_editor_head_pitch')
	portrait_editor_head_yaw = state_manager.get_item('portrait_editor_head_yaw')
	portrait_editor_head_roll = state_manager.get_item('portrait_editor_head_roll')
	edit_pitch = pitch + float(numpy.interp(portrait_editor_head_pitch, [ -1, 1 ], [ 20, -20 ]))
	edit_yaw = yaw + float(numpy.interp(portrait_editor_head_yaw, [ -1, 1 ], [ 60, -60 ]))
	edit_roll = roll + float(numpy.interp(portrait_editor_head_roll, [ -1, 1 ], [ -15, 15 ]))
	edit_pitch, edit_yaw, edit_roll = limit_angle(pitch, yaw, roll, edit_pitch, edit_yaw, edit_roll)
	rotation = create_rotation(edit_pitch, edit_yaw, edit_roll)
	return rotation


def calculate_distance_ratio(region_landmark_68 : RegionLandmark68, top_index : int, bottom_index : int, left_index : int, right_index : int) -> float:
	vertical_direction = region_landmark_68[top_index] - region_landmark_68[bottom_index]
	horizontal_direction = region_landmark_68[left_index] - region_landmark_68[right_index]
	distance_ratio = float(numpy.linalg.norm(vertical_direction) / (numpy.linalg.norm(horizontal_direction) + 1e-6))
	return distance_ratio


def prepare_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	model_size = get_model_options().get('size')
	prepare_size = (model_size[0] // 2, model_size[1] // 2)
	crop_vision_frame = cv2.resize(crop_vision_frame, prepare_size, interpolation = cv2.INTER_AREA)
	crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
	crop_vision_frame = numpy.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
	return crop_vision_frame


def normalize_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = crop_vision_frame.transpose(1, 2, 0).clip(0, 1)
	crop_vision_frame = (crop_vision_frame * 255.0)
	crop_vision_frame = crop_vision_frame.astype(numpy.uint8)[:, :, ::-1]
	return crop_vision_frame


def process_frame(inputs : PortraitEditorInputs) -> ProcessorOutputs:
	reference_vision_frame = inputs.get('reference_vision_frame')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	target_regions = select_regions(reference_vision_frame, target_vision_frame)

	if target_regions:
		for target_region in target_regions:
			target_region = scale_region(target_region, target_vision_frame, temp_vision_frame)
			temp_vision_frame = edit_portrait(target_region, temp_vision_frame)

	return temp_vision_frame, temp_vision_mask
