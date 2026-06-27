from functools import lru_cache
from typing import Tuple

import numpy

from videokit import inference_manager
from videokit.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from videokit.region_helper import warp_region_by_region_landmark_5
from videokit.filesystem import resolve_relative_path
from videokit.thread_helper import conditional_thread_semaphore
from videokit.types import DownloadScope, Embedding, RegionLandmark5, InferencePool, ModelOptions, ModelSet, VisionFrame


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'arcface':
		{
			'__metadata__':
			{
				'vendor': 'InsightFace',
				'license': 'Non-Commercial',
				'year': 2018
			},
			'hashes':
			{
				'region_recognizer':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_w600k_r50.hash'),
					'path': resolve_relative_path('../.assets/models/vk_2c3d.hash')
				}
			},
			'sources':
			{
				'region_recognizer':
				{
					'url': resolve_download_url('models-3.0.0', 'arcface_w600k_r50.onnx'),
					'path': resolve_relative_path('../.assets/models/vk_2c3d.onnx')
				}
			},
			'template': 'arcface_112_v2',
			'size': (112, 112)
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ 'arcface' ]
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ 'arcface' ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
	return create_static_model_set('full').get('arcface')


def pre_check() -> bool:
	model_hash_set = get_model_options().get('hashes')
	model_source_set = get_model_options().get('sources')

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def calculate_region_embedding(temp_vision_frame : VisionFrame, region_landmark_5 : RegionLandmark5) -> Tuple[Embedding, Embedding]:
	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	crop_vision_frame, matrix = warp_region_by_region_landmark_5(temp_vision_frame, region_landmark_5, model_template, model_size)
	crop_vision_frame = crop_vision_frame / 127.5 - 1
	crop_vision_frame = crop_vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
	crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis = 0)
	region_embedding = forward(crop_vision_frame)
	region_embedding = region_embedding.ravel()
	region_embedding_norm = region_embedding / numpy.linalg.norm(region_embedding)
	return region_embedding, region_embedding_norm


def forward(crop_vision_frame : VisionFrame) -> Embedding:
	region_recognizer = get_inference_pool().get('region_recognizer')

	with conditional_thread_semaphore():
		region_embedding = region_recognizer.run(None,
		{
			'input': crop_vision_frame
		})[0]

	return region_embedding
