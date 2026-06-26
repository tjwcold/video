from argparse import ArgumentParser
from functools import lru_cache
from typing import List, Optional, Tuple

import cv2
import numpy

import videokit.choices
import videokit.jobs.job_manager
import videokit.jobs.job_store
from videokit import config, content_analyser, region_classifier, region_detector, landmark_detector, region_masker, region_recognizer, inference_manager, logger, state_manager, translator, video_manager
from videokit.common_helper import get_first, is_macos
from videokit.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from videokit.execution import has_execution_provider
from videokit.region_analyser import get_average_region, get_many_regions, get_one_region, scale_region
from videokit.region_helper import paste_back, warp_region_by_region_landmark_5
from videokit.region_masker import create_area_mask, create_box_mask, create_occlusion_mask, create_region_mask
from videokit.region_selector import select_regions, sort_regions_by_order
from videokit.filesystem import filter_image_paths, has_image, in_directory, is_file, is_image, is_video, resolve_relative_path, same_file_extension
from videokit.model_helper import get_static_model_initializer
from videokit.processors.modules.style_transfer import choices as style_transfer_choices
from videokit.processors.modules.style_transfer.types import StyleTransferInputs
from videokit.processors.pixel_boost import explode_pixel_boost, implode_pixel_boost
from videokit.processors.types import ProcessorOutputs
from videokit.program_helper import find_argument_group
from videokit.thread_helper import conditional_thread_semaphore
from videokit.types import ApplyStateItem, Args, DownloadScope, Embedding, Region, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from videokit.vision import read_static_image, read_static_images, read_static_video_frame, unpack_resolution


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'blendswap_256':
		{
			'__metadata__':
			{
				'vendor': 'mapooon',
				'license': 'Non-Commercial',
				'year': 2023
			},
			'hashes':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'blendswap_256.hash'),
					'path': resolve_relative_path('../.assets/models/blendswap_256.hash')
				}
			},
			'sources':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'blendswap_256.onnx'),
					'path': resolve_relative_path('../.assets/models/blendswap_256.onnx')
				}
			},
			'type': 'blendswap',
			'template': 'ffhq_512',
			'size': (256, 256),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'ghost_1_256':
		{
			'__metadata__':
			{
				'vendor': 'ai-forever',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_1_256.hash'),
					'path': resolve_relative_path('../.assets/models/ghost_1_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.hash')
				}
			},
			'sources':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_1_256.onnx'),
					'path': resolve_relative_path('../.assets/models/ghost_1_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.onnx')
				}
			},
			'type': 'ghost',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'ghost_2_256':
		{
			'__metadata__':
			{
				'vendor': 'ai-forever',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_2_256.hash'),
					'path': resolve_relative_path('../.assets/models/ghost_2_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.hash')
				}
			},
			'sources':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_2_256.onnx'),
					'path': resolve_relative_path('../.assets/models/ghost_2_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.onnx')
				}
			},
			'type': 'ghost',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'ghost_3_256':
		{
			'__metadata__':
			{
				'vendor': 'ai-forever',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_3_256.hash'),
					'path': resolve_relative_path('../.assets/models/ghost_3_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.hash')
				}
			},
			'sources':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_3_256.onnx'),
					'path': resolve_relative_path('../.assets/models/ghost_3_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.onnx')
				}
			},
			'type': 'ghost',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hififace_unofficial_256':
		{
			'__metadata__':
			{
				'vendor': 'GuijiAI',
				'license': 'Unknown',
				'year': 2021
			},
			'hashes':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.1.0', 'hififace_unofficial_256.hash'),
					'path': resolve_relative_path('../.assets/models/hififace_unofficial_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_hififace.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_hififace.hash')
				}
			},
			'sources':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.1.0', 'hififace_unofficial_256.onnx'),
					'path': resolve_relative_path('../.assets/models/hififace_unofficial_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_hififace.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_hififace.onnx')
				}
			},
			'type': 'hififace',
			'template': 'mtcnn_512',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hyperswap_1a_256':
		{
			'__metadata__':
			{
				'vendor': 'VideoKit',
				'license': 'ResearchRAIL',
				'year': 2025
			},
			'hashes':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1a_256.hash'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1a_256.hash')
				}
			},
			'sources':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1a_256.onnx'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1a_256.onnx')
				}
			},
			'type': 'hyperswap',
			'template': 'arcface_128',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hyperswap_1b_256':
		{
			'__metadata__':
			{
				'vendor': 'VideoKit',
				'license': 'ResearchRAIL',
				'year': 2025
			},
			'hashes':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1b_256.hash'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1b_256.hash')
				}
			},
			'sources':
				{
					'style_transfer':
					{
						'url': resolve_download_url('models-3.3.0', 'hyperswap_1b_256.onnx'),
						'path': resolve_relative_path('../.assets/models/hyperswap_1b_256.onnx')
					}
				},
			'type': 'hyperswap',
			'template': 'arcface_128',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hyperswap_1c_256':
		{
			'__metadata__':
			{
				'vendor': 'VideoKit',
				'license': 'ResearchRAIL',
				'year': 2025
			},
			'hashes':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1c_256.hash'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1c_256.hash')
				}
			},
			'sources':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1c_256.onnx'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1c_256.onnx')
				}
			},
			'type': 'hyperswap',
			'template': 'arcface_128',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'inswapper_128':
		{
			'__metadata__':
			{
				'vendor': 'InsightFace',
				'license': 'Non-Commercial',
				'year': 2023
			},
			'hashes':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128.hash'),
					'path': resolve_relative_path('../.assets/models/inswapper_128.hash')
				}
			},
			'sources':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128.onnx'),
					'path': resolve_relative_path('../.assets/models/inswapper_128.onnx')
				}
			},
			'type': 'inswapper',
			'template': 'arcface_128',
			'size': (128, 128),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'inswapper_128_fp16':
		{
			'__metadata__':
			{
				'vendor': 'InsightFace',
				'license': 'Non-Commercial',
				'year': 2023
			},
			'hashes':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128_fp16.hash'),
					'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.hash')
				}
			},
			'sources':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128_fp16.onnx'),
					'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.onnx')
				}
			},
			'type': 'inswapper',
			'template': 'arcface_128',
			'size': (128, 128),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'simswap_256':
		{
			'__metadata__':
			{
				'vendor': 'neuralchen',
				'license': 'Non-Commercial',
				'year': 2020
			},
			'hashes':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_256.hash'),
					'path': resolve_relative_path('../.assets/models/simswap_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_simswap.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_simswap.hash')
				}
			},
			'sources':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_256.onnx'),
					'path': resolve_relative_path('../.assets/models/simswap_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_simswap.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_simswap.onnx')
				}
			},
			'type': 'simswap',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.485, 0.456, 0.406 ],
			'standard_deviation': [ 0.229, 0.224, 0.225 ]
		},
		'simswap_unofficial_512':
		{
			'__metadata__':
			{
				'vendor': 'neuralchen',
				'license': 'Non-Commercial',
				'year': 2020
			},
			'hashes':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_unofficial_512.hash'),
					'path': resolve_relative_path('../.assets/models/simswap_unofficial_512.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_simswap.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_simswap.hash')
				}
			},
			'sources':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_unofficial_512.onnx'),
					'path': resolve_relative_path('../.assets/models/simswap_unofficial_512.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_simswap.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_simswap.onnx')
				}
			},
			'type': 'simswap',
			'template': 'arcface_112_v1',
			'size': (512, 512),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'uniface_256':
		{
			'__metadata__':
			{
				'vendor': 'xc-csc101',
				'license': 'Unknown',
				'year': 2022
			},
			'hashes':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'uniface_256.hash'),
					'path': resolve_relative_path('../.assets/models/uniface_256.hash')
				}
			},
			'sources':
			{
				'style_transfer':
				{
					'url': resolve_download_url('models-3.0.0', 'uniface_256.onnx'),
					'path': resolve_relative_path('../.assets/models/uniface_256.onnx')
				}
			},
			'type': 'uniface',
			'template': 'ffhq_512',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ get_model_name() ]
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ get_model_name() ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
	model_name = get_model_name()
	return create_static_model_set('full').get(model_name)


def get_model_name() -> str:
	model_name = state_manager.get_item('style_transfer_model') or 'inswapper_128'

	if is_macos() and has_execution_provider('coreml') and model_name == 'inswapper_128_fp16':
		return 'inswapper_128'
	return model_name


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--style-transfer-model', help = translator.get('help.model', __package__), default = config.get_str_value('processors', 'style_transfer_model', 'hyperswap_1a_256'), choices = style_transfer_choices.style_transfer_models)
		known_args, _ = program.parse_known_args()
		style_transfer_pixel_boost_choices = style_transfer_choices.style_transfer_set.get(known_args.style_transfer_model)
		group_processors.add_argument('--style-transfer-pixel-boost', help = translator.get('help.pixel_boost', __package__), default = config.get_str_value('processors', 'style_transfer_pixel_boost', get_first(style_transfer_pixel_boost_choices)), choices = style_transfer_pixel_boost_choices)
		group_processors.add_argument('--style-transfer-weight', help = translator.get('help.weight', __package__), type = float, default = config.get_float_value('processors', 'style_transfer_weight', '0.5'), choices = style_transfer_choices.style_transfer_weight_range)
		videokit.jobs.job_store.register_step_keys([ 'style_transfer_model', 'style_transfer_pixel_boost', 'style_transfer_weight' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('style_transfer_model', args.get('style_transfer_model'))
	apply_state_item('style_transfer_pixel_boost', args.get('style_transfer_pixel_boost'))
	apply_state_item('style_transfer_weight', args.get('style_transfer_weight'))


def pre_check() -> bool:
	return True


def pre_process(mode : ProcessMode) -> bool:
	model_options = get_model_options()
	if not model_options:
		return False
	model_source_set = model_options.get('sources')
	if not model_source_set:
		return False
	for model_name in model_source_set.keys():
		model_path = model_source_set.get(model_name).get('path')
		if not is_file(model_path):
			return False
	return True


def post_process() -> None:
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()
	video_manager.clear_video_pool()
	if state_manager.get_item('video_memory_strategy') in [ 'strict', 'moderate' ]:
		get_static_model_initializer.cache_clear()
		clear_inference_pool()
	if state_manager.get_item('video_memory_strategy') == 'strict':
		content_analyser.clear_inference_pool()
		region_classifier.clear_inference_pool()
		region_detector.clear_inference_pool()
		landmark_detector.clear_inference_pool()
		region_masker.clear_inference_pool()
		region_recognizer.clear_inference_pool()


def transfer_style(source_region : Region, target_region : Region, temp_vision_frame : VisionFrame) -> VisionFrame:
	model_options = get_model_options()
	if not model_options:
		return temp_vision_frame
	style_transfer_model = get_inference_pool().get('style_transfer')
	if not style_transfer_model:
		return temp_vision_frame
	model_template = model_options.get('template')
	model_size = model_options.get('size')
	pixel_boost_size = unpack_resolution(state_manager.get_item('style_transfer_pixel_boost'))
	pixel_boost_total = pixel_boost_size[0] // model_size[0]
	crop_vision_frame, affine_matrix = warp_region_by_region_landmark_5(temp_vision_frame, target_region.landmark_set.get('5/68'), model_template, pixel_boost_size)
	temp_vision_frames = []
	crop_masks = []

	if 'box' in state_manager.get_item('region_mask_types'):
		box_mask = create_box_mask(crop_vision_frame, state_manager.get_item('region_mask_blur'), state_manager.get_item('region_mask_padding'))
		crop_masks.append(box_mask)

	if 'occlusion' in state_manager.get_item('region_mask_types'):
		occlusion_mask = create_occlusion_mask(crop_vision_frame)
		crop_masks.append(occlusion_mask)

	pixel_boost_vision_frames = implode_pixel_boost(crop_vision_frame, pixel_boost_total, model_size)
	for pixel_boost_vision_frame in pixel_boost_vision_frames:
		pixel_boost_vision_frame = prepare_crop_frame(pixel_boost_vision_frame)
		pixel_boost_vision_frame = forward_transfer_style(source_region, target_region, pixel_boost_vision_frame)
		pixel_boost_vision_frame = normalize_crop_frame(pixel_boost_vision_frame)
		temp_vision_frames.append(pixel_boost_vision_frame)
	crop_vision_frame = explode_pixel_boost(temp_vision_frames, pixel_boost_total, model_size, pixel_boost_size)

	if 'area' in state_manager.get_item('region_mask_types'):
		region_landmark_68 = cv2.transform(target_region.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
		area_mask = create_area_mask(crop_vision_frame, region_landmark_68, state_manager.get_item('region_mask_areas'))
		crop_masks.append(area_mask)

	if 'region' in state_manager.get_item('region_mask_types'):
		region_mask = create_region_mask(crop_vision_frame, state_manager.get_item('region_mask_regions'))
		crop_masks.append(region_mask)

	crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
	paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
	return paste_vision_frame


def forward_transfer_style(source_region : Region, target_region : Region, crop_vision_frame : VisionFrame) -> VisionFrame:
	style_transfer = get_inference_pool().get('style_transfer')
	if not style_transfer:
		return crop_vision_frame
	model_type = get_model_options().get('type')
	style_transfer_inputs = {}

	if is_macos() and has_execution_provider('coreml') and model_type in [ 'ghost', 'uniface' ]:
		style_transfer.set_providers([ videokit.choices.execution_provider_set.get('cpu') ])

	for style_transfer_input in style_transfer.get_inputs():
		if style_transfer_input.name == 'source':
			if model_type in [ 'blendswap', 'uniface' ]:
				style_transfer_inputs[style_transfer_input.name] = prepare_source_frame(source_region)
			else:
				source_embedding = prepare_source_embedding(source_region)
				source_embedding = balance_source_embedding(source_embedding, target_region.embedding)
				style_transfer_inputs[style_transfer_input.name] = source_embedding
		if style_transfer_input.name == 'target':
			style_transfer_inputs[style_transfer_input.name] = crop_vision_frame

	with conditional_thread_semaphore():
		crop_vision_frame = style_transfer.run(None, style_transfer_inputs)[0][0]

	return crop_vision_frame


def forward_convert_embedding(region_embedding : Embedding) -> Embedding:
	embedding_converter = get_inference_pool().get('embedding_converter')
	if not embedding_converter:
		return region_embedding

	with conditional_thread_semaphore():
		region_embedding = embedding_converter.run(None,
		{
			'input': region_embedding
		})[0]

	return region_embedding


def prepare_source_frame(source_region : Region) -> VisionFrame:
	model_type = get_model_options().get('type')
	source_vision_frame = read_static_image(get_first(state_manager.get_item('source_paths')))

	if model_type == 'blendswap':
		source_vision_frame, _ = warp_region_by_region_landmark_5(source_vision_frame, source_region.landmark_set.get('5/68'), 'arcface_112_v2', (112, 112))

	if model_type == 'uniface':
		source_vision_frame, _ = warp_region_by_region_landmark_5(source_vision_frame, source_region.landmark_set.get('5/68'), 'ffhq_512', (256, 256))

	source_vision_frame = source_vision_frame[:, :, ::-1] / 255.0
	source_vision_frame = source_vision_frame.transpose(2, 0, 1)
	source_vision_frame = numpy.expand_dims(source_vision_frame, axis = 0).astype(numpy.float32)
	return source_vision_frame


def prepare_source_embedding(source_region : Region) -> Embedding:
	model_type = get_model_options().get('type')

	if model_type == 'ghost':
		source_embedding = source_region.embedding.reshape(-1, 512)
		source_embedding, _ = convert_source_embedding(source_embedding)
		source_embedding = source_embedding.reshape(1, -1)
		return source_embedding

	if model_type == 'hyperswap':
		source_embedding = source_region.embedding_norm.reshape((1, -1))
		return source_embedding

	if model_type == 'inswapper':
		model_path = get_model_options().get('sources').get('style_transfer').get('path')
		model_initializer = get_static_model_initializer(model_path)
		source_embedding = source_region.embedding.reshape((1, -1))
		source_embedding = numpy.dot(source_embedding, model_initializer) / numpy.linalg.norm(source_embedding)
		return source_embedding

	source_embedding = source_region.embedding.reshape(-1, 512)
	_, source_embedding_norm = convert_source_embedding(source_embedding)
	source_embedding = source_embedding_norm.reshape(1, -1)
	return source_embedding


def balance_source_embedding(source_embedding : Embedding, target_embedding : Embedding) -> Embedding:
	model_type = get_model_options().get('type')
	style_transfer_weight = state_manager.get_item('style_transfer_weight')
	style_transfer_weight = numpy.interp(style_transfer_weight, [ 0, 1 ], [ 0.35, -0.35 ]).astype(numpy.float32)

	if model_type in [ 'hififace', 'hyperswap', 'inswapper', 'simswap' ]:
		target_embedding = target_embedding / numpy.linalg.norm(target_embedding)

	source_embedding = source_embedding.reshape(1, -1)
	target_embedding = target_embedding.reshape(1, -1)
	source_embedding = source_embedding * (1 - style_transfer_weight) + target_embedding * style_transfer_weight
	return source_embedding


def convert_source_embedding(source_embedding : Embedding) -> Tuple[Embedding, Embedding]:
	source_embedding = forward_convert_embedding(source_embedding)
	source_embedding = source_embedding.ravel()
	source_embedding_norm = source_embedding / numpy.linalg.norm(source_embedding)
	return source_embedding, source_embedding_norm


def prepare_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	model_mean = get_model_options().get('mean')
	model_standard_deviation = get_model_options().get('standard_deviation')

	crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
	crop_vision_frame = (crop_vision_frame - model_mean) / model_standard_deviation
	crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
	crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis = 0).astype(numpy.float32)
	return crop_vision_frame


def normalize_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	model_type = get_model_options().get('type')
	model_mean = get_model_options().get('mean')
	model_standard_deviation = get_model_options().get('standard_deviation')

	crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)

	if model_type in [ 'ghost', 'hififace', 'hyperswap', 'uniface' ]:
		crop_vision_frame = crop_vision_frame * model_standard_deviation + model_mean

	crop_vision_frame = crop_vision_frame.clip(0, 1)
	crop_vision_frame = crop_vision_frame[:, :, ::-1] * 255
	return crop_vision_frame


def extract_source_region(source_vision_frames : List[VisionFrame]) -> Optional[Region]:
	source_regions = []

	if source_vision_frames:
		for source_vision_frame in source_vision_frames:
			temp_regions = get_many_regions([source_vision_frame])
			temp_regions = sort_regions_by_order(temp_regions, 'large-small')

			if temp_regions:
				source_regions.append(get_first(temp_regions))

	return get_average_region(source_regions)


def process_frame(inputs : StyleTransferInputs) -> ProcessorOutputs:
	reference_vision_frame = inputs.get('reference_vision_frame')
	source_vision_frames = inputs.get('source_vision_frames')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	source_region = extract_source_region(source_vision_frames)
	target_regions = select_regions(reference_vision_frame, target_vision_frame)

	if source_region and target_regions:
		for target_region in target_regions:
			target_region = scale_region(target_region, target_vision_frame, temp_vision_frame)
			temp_vision_frame = transfer_style(source_region, target_region, temp_vision_frame)

	return temp_vision_frame, temp_vision_mask
