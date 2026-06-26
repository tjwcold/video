from argparse import ArgumentParser
from functools import lru_cache

import numpy

import videokit.jobs.job_manager
import videokit.jobs.job_store
from videokit import config, content_analyser, region_classifier, region_detector, landmark_detector, region_masker, region_recognizer, inference_manager, logger, state_manager, translator, video_manager
from videokit.common_helper import create_float_metavar, create_int_metavar
from videokit.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from videokit.region_analyser import scale_region
from videokit.region_helper import paste_back, warp_region_by_region_landmark_5
from videokit.region_masker import create_box_mask, create_occlusion_mask
from videokit.region_selector import select_regions
from videokit.filesystem import in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from videokit.processors.modules.quality_enhancer import choices as quality_enhancer_choices
from videokit.processors.modules.quality_enhancer.types import QualityEnhancerInputs, QualityEnhancerWeight
from videokit.processors.types import ProcessorOutputs
from videokit.program_helper import find_argument_group
from videokit.thread_helper import thread_semaphore
from videokit.types import ApplyStateItem, Args, DownloadScope, Region, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from videokit.vision import blend_frame, read_static_image, read_static_video_frame


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'codeformer':
		{
			'__metadata__':
			{
				'vendor': 'sczhou',
				'license': 'S-Lab-1.0',
				'year': 2022
			},
			'hashes':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'codeformer.hash'),
					'path': resolve_relative_path('../.assets/models/codeformer.hash')
				}
			},
			'sources':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'codeformer.onnx'),
					'path': resolve_relative_path('../.assets/models/codeformer.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		},
		'gfpgan_1.2':
		{
			'__metadata__':
			{
				'vendor': 'TencentARC',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.2.hash'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.2.hash')
				}
			},
			'sources':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.2.onnx'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.2.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		},
		'gfpgan_1.3':
		{
			'__metadata__':
			{
				'vendor': 'TencentARC',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.3.hash'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.3.hash')
				}
			},
			'sources':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.3.onnx'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.3.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		},
		'gfpgan_1.4':
		{
			'__metadata__':
			{
				'vendor': 'TencentARC',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.4.hash'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.4.hash')
				}
			},
			'sources':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gfpgan_1.4.onnx'),
					'path': resolve_relative_path('../.assets/models/gfpgan_1.4.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		},
		'gpen_bfr_256':
		{
			'__metadata__':
			{
				'vendor': 'yangxy',
				'license': 'Non-Commercial',
				'year': 2021
			},
			'hashes':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_256.hash'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_256.hash')
				}
			},
			'sources':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_256.onnx'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_256.onnx')
				}
			},
			'template': 'arcface_128',
			'size': (256, 256)
		},
		'gpen_bfr_512':
		{
			'__metadata__':
			{
				'vendor': 'yangxy',
				'license': 'Non-Commercial',
				'year': 2021
			},
			'hashes':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_512.hash'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_512.hash')
				}
			},
			'sources':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_512.onnx'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_512.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		},
		'gpen_bfr_1024':
		{
			'__metadata__':
			{
				'vendor': 'yangxy',
				'license': 'Non-Commercial',
				'year': 2021
			},
			'hashes':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_1024.hash'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_1024.hash')
				}
			},
			'sources':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_1024.onnx'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_1024.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (1024, 1024)
		},
		'gpen_bfr_2048':
		{
			'__metadata__':
			{
				'vendor': 'yangxy',
				'license': 'Non-Commercial',
				'year': 2021
			},
			'hashes':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_2048.hash'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_2048.hash')
				}
			},
			'sources':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'gpen_bfr_2048.onnx'),
					'path': resolve_relative_path('../.assets/models/gpen_bfr_2048.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (2048, 2048)
		},
		'restoreformer_plus_plus':
		{
			'__metadata__':
			{
				'vendor': 'wzhouxiff',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'restoreformer_plus_plus.hash'),
					'path': resolve_relative_path('../.assets/models/restoreformer_plus_plus.hash')
				}
			},
			'sources':
			{
				'quality_enhancer':
				{
					'url': resolve_download_url('models-3.0.0', 'restoreformer_plus_plus.onnx'),
					'path': resolve_relative_path('../.assets/models/restoreformer_plus_plus.onnx')
				}
			},
			'template': 'ffhq_512',
			'size': (512, 512)
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ state_manager.get_item('quality_enhancer_model') ]
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ state_manager.get_item('quality_enhancer_model') ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
	model_name = state_manager.get_item('quality_enhancer_model')
	return create_static_model_set('full').get(model_name)


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--quality-enhancer-model', help = translator.get('help.model', __package__), default = config.get_str_value('processors', 'quality_enhancer_model', 'gfpgan_1.4'), choices = quality_enhancer_choices.quality_enhancer_models)
		group_processors.add_argument('--quality-enhancer-blend', help = translator.get('help.blend', __package__), type = int, default = config.get_int_value('processors', 'quality_enhancer_blend', '80'), choices = quality_enhancer_choices.quality_enhancer_blend_range, metavar = create_int_metavar(quality_enhancer_choices.quality_enhancer_blend_range))
		group_processors.add_argument('--quality-enhancer-weight', help = translator.get('help.weight', __package__), type = float, default = config.get_float_value('processors', 'quality_enhancer_weight', '0.5'), choices = quality_enhancer_choices.quality_enhancer_weight_range, metavar = create_float_metavar(quality_enhancer_choices.quality_enhancer_weight_range))
		videokit.jobs.job_store.register_step_keys([ 'quality_enhancer_model', 'quality_enhancer_blend', 'quality_enhancer_weight' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('quality_enhancer_model', args.get('quality_enhancer_model'))
	apply_state_item('quality_enhancer_blend', args.get('quality_enhancer_blend'))
	apply_state_item('quality_enhancer_weight', args.get('quality_enhancer_weight'))


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


def enhance_region(target_region : Region, temp_vision_frame : VisionFrame) -> VisionFrame:
	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	crop_vision_frame, affine_matrix = warp_region_by_region_landmark_5(temp_vision_frame, target_region.landmark_set.get('5/68'), model_template, model_size)
	box_mask = create_box_mask(crop_vision_frame, state_manager.get_item('region_mask_blur'), (0, 0, 0, 0))
	crop_masks =\
	[
		box_mask
	]

	if 'occlusion' in state_manager.get_item('region_mask_types'):
		occlusion_mask = create_occlusion_mask(crop_vision_frame)
		crop_masks.append(occlusion_mask)

	crop_vision_frame = prepare_crop_frame(crop_vision_frame)
	quality_enhancer_weight = numpy.array([ state_manager.get_item('quality_enhancer_weight') ]).astype(numpy.double)
	crop_vision_frame = forward(crop_vision_frame, quality_enhancer_weight)
	crop_vision_frame = normalize_crop_frame(crop_vision_frame)
	crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
	paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
	temp_vision_frame = blend_paste_frame(temp_vision_frame, paste_vision_frame)
	return temp_vision_frame


def forward(crop_vision_frame : VisionFrame, quality_enhancer_weight : QualityEnhancerWeight) -> VisionFrame:
	quality_enhancer = get_inference_pool().get('quality_enhancer')
	quality_enhancer_inputs = {}

	for quality_enhancer_input in quality_enhancer.get_inputs():
		if quality_enhancer_input.name == 'input':
			quality_enhancer_inputs[quality_enhancer_input.name] = crop_vision_frame
		if quality_enhancer_input.name == 'weight':
			quality_enhancer_inputs[quality_enhancer_input.name] = quality_enhancer_weight

	with thread_semaphore():
		crop_vision_frame = quality_enhancer.run(None, quality_enhancer_inputs)[0][0]

	return crop_vision_frame


def has_weight_input() -> bool:
	quality_enhancer = get_inference_pool().get('quality_enhancer')

	for deep_swapper_input in quality_enhancer.get_inputs():
		if deep_swapper_input.name == 'weight':
			return True

	return False


def prepare_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
	crop_vision_frame = (crop_vision_frame - 0.5) / 0.5
	crop_vision_frame = numpy.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
	return crop_vision_frame


def normalize_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = numpy.clip(crop_vision_frame, -1, 1)
	crop_vision_frame = (crop_vision_frame + 1) / 2
	crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
	crop_vision_frame = (crop_vision_frame * 255.0).round()
	crop_vision_frame = crop_vision_frame.astype(numpy.uint8)[:, :, ::-1]
	return crop_vision_frame


def blend_paste_frame(temp_vision_frame : VisionFrame, paste_vision_frame : VisionFrame) -> VisionFrame:
	quality_enhancer_blend = 1 - (state_manager.get_item('quality_enhancer_blend') / 100)
	temp_vision_frame = blend_frame(temp_vision_frame, paste_vision_frame, 1 - quality_enhancer_blend)
	return temp_vision_frame


def process_frame(inputs : QualityEnhancerInputs) -> ProcessorOutputs:
	reference_vision_frame = inputs.get('reference_vision_frame')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	target_regions = select_regions(reference_vision_frame, target_vision_frame)

	if target_regions:
		for target_region in target_regions:
			target_region = scale_region(target_region, target_vision_frame, temp_vision_frame)
			temp_vision_frame = enhance_region(target_region, temp_vision_frame)

	return temp_vision_frame, temp_vision_mask
