from videokit import state_manager
from videokit.filesystem import get_file_name, is_video, resolve_file_paths
from videokit.jobs import job_store
from videokit.normalizer import normalize_fps, normalize_space
from videokit.processors.core import get_processors_modules
from videokit.types import ApplyStateItem, Args
from videokit.vision import detect_video_fps


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('command', args.get('command'))
	apply_state_item('temp_path', args.get('temp_path'))
	apply_state_item('jobs_path', args.get('jobs_path'))
	apply_state_item('source_paths', args.get('source_paths'))
	apply_state_item('target_path', args.get('target_path'))
	apply_state_item('output_path', args.get('output_path'))
	apply_state_item('source_pattern', args.get('source_pattern'))
	apply_state_item('target_pattern', args.get('target_pattern'))
	apply_state_item('output_pattern', args.get('output_pattern'))
	apply_state_item('region_detector_model', args.get('region_detector_model'))
	apply_state_item('region_detector_size', args.get('region_detector_size'))
	apply_state_item('region_detector_margin', normalize_space(args.get('region_detector_margin')))
	apply_state_item('region_detector_angles', args.get('region_detector_angles'))
	apply_state_item('region_detector_score', args.get('region_detector_score'))
	apply_state_item('landmark_detector_model', args.get('landmark_detector_model'))
	apply_state_item('landmark_detector_score', args.get('landmark_detector_score'))
	apply_state_item('region_selector_mode', args.get('region_selector_mode'))
	apply_state_item('region_selector_order', args.get('region_selector_order'))
	apply_state_item('region_selector_age_start', args.get('region_selector_age_start'))
	apply_state_item('region_selector_age_end', args.get('region_selector_age_end'))
	apply_state_item('region_selector_gender', args.get('region_selector_gender'))
	apply_state_item('region_selector_race', args.get('region_selector_race'))
	apply_state_item('reference_region_position', args.get('reference_region_position'))
	apply_state_item('reference_region_distance', args.get('reference_region_distance'))
	apply_state_item('reference_frame_number', args.get('reference_frame_number'))
	apply_state_item('region_occluder_model', args.get('region_occluder_model'))
	apply_state_item('region_parser_model', args.get('region_parser_model'))
	apply_state_item('region_mask_types', args.get('region_mask_types'))
	apply_state_item('region_mask_areas', args.get('region_mask_areas'))
	apply_state_item('region_mask_regions', args.get('region_mask_regions'))
	apply_state_item('region_mask_blur', args.get('region_mask_blur'))
	apply_state_item('region_mask_padding', normalize_space(args.get('region_mask_padding')))
	apply_state_item('voice_extractor_model', args.get('voice_extractor_model'))
	apply_state_item('trim_frame_start', args.get('trim_frame_start'))
	apply_state_item('trim_frame_end', args.get('trim_frame_end'))
	apply_state_item('temp_frame_format', args.get('temp_frame_format'))
	apply_state_item('keep_temp', args.get('keep_temp'))
	apply_state_item('output_image_quality', args.get('output_image_quality'))
	apply_state_item('output_image_scale', args.get('output_image_scale'))
	apply_state_item('output_audio_encoder', args.get('output_audio_encoder'))
	apply_state_item('output_audio_quality', args.get('output_audio_quality'))
	apply_state_item('output_audio_volume', args.get('output_audio_volume'))
	apply_state_item('output_video_encoder', args.get('output_video_encoder'))
	apply_state_item('output_video_preset', args.get('output_video_preset'))
	apply_state_item('output_video_quality', args.get('output_video_quality'))
	apply_state_item('output_video_scale', args.get('output_video_scale'))

	if args.get('output_video_fps') or is_video(args.get('target_path')):
		output_video_fps = normalize_fps(args.get('output_video_fps')) or detect_video_fps(args.get('target_path'))
		apply_state_item('output_video_fps', output_video_fps)

	available_processors = [ get_file_name(file_path) for file_path in resolve_file_paths('videokit/processors/modules') ]
	apply_state_item('processors', args.get('processors'))

	for processor_module in get_processors_modules(available_processors):
		processor_module.apply_args(args, apply_state_item)

	apply_state_item('open_browser', args.get('open_browser'))
	apply_state_item('ui_layouts', args.get('ui_layouts'))
	apply_state_item('ui_workflow', args.get('ui_workflow'))
	apply_state_item('execution_device_ids', args.get('execution_device_ids'))
	apply_state_item('execution_providers', args.get('execution_providers'))
	apply_state_item('execution_thread_count', args.get('execution_thread_count'))
	apply_state_item('download_providers', args.get('download_providers'))
	apply_state_item('download_scope', args.get('download_scope'))
	apply_state_item('benchmark_mode', args.get('benchmark_mode'))
	apply_state_item('benchmark_resolutions', args.get('benchmark_resolutions'))
	apply_state_item('benchmark_cycle_count', args.get('benchmark_cycle_count'))
	apply_state_item('video_memory_strategy', args.get('video_memory_strategy'))
	apply_state_item('system_memory_limit', args.get('system_memory_limit'))
	apply_state_item('log_level', args.get('log_level'))
	apply_state_item('halt_on_error', args.get('halt_on_error'))
	apply_state_item('job_id', args.get('job_id'))
	apply_state_item('job_status', args.get('job_status'))
	apply_state_item('step_index', args.get('step_index'))


def reduce_step_args(args : Args) -> Args:
	step_args =\
	{
		key: args[key] for key in args if key in job_store.get_step_keys()
	}
	return step_args


def reduce_job_args(args : Args) -> Args:
	job_args =\
	{
		key: args[key] for key in args if key in job_store.get_job_keys()
	}
	return job_args


def collect_step_args() -> Args:
	step_args =\
	{
		key: state_manager.get_item(key) for key in job_store.get_step_keys() #type:ignore[arg-type]
	}
	return step_args


def collect_job_args() -> Args:
	job_args =\
	{
		key: state_manager.get_item(key) for key in job_store.get_job_keys() #type:ignore[arg-type]
	}
	return job_args
