from functools import lru_cache
from typing import Tuple

import cv2
import numpy

from videokit import inference_manager, state_manager
from videokit.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from videokit.region_helper import create_rotation_matrix_and_size, estimate_matrix_by_region_landmark_5, transform_points, warp_region_by_translation
from videokit.filesystem import resolve_relative_path
from videokit.thread_helper import conditional_thread_semaphore
from videokit.types import Angle, BoundingBox, DownloadScope, DownloadSet, RegionLandmark5, RegionLandmark68, InferencePool, ModelSet, Prediction, Score, VisionFrame


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'2dfan4':
		{
			'__metadata__':
			{
				'vendor': 'breadbread1984',
				'license': 'MIT',
				'year': 2018
			},
			'hashes':
			{
				'2dfan4':
				{
					'url': resolve_download_url('models-3.0.0', '2dfan4.hash'),
					'path': resolve_relative_path('../.assets/models/2dfan4.hash')
				}
			},
			'sources':
			{
				'2dfan4':
				{
					'url': resolve_download_url('models-3.0.0', '2dfan4.onnx'),
					'path': resolve_relative_path('../.assets/models/2dfan4.onnx')
				}
			},
			'size': (256, 256)
		},
		'peppa_wutz':
		{
			'__metadata__':
			{
				'vendor': 'Unknown',
				'license': 'Apache-2.0',
				'year': 2023
			},
			'hashes':
			{
				'peppa_wutz':
				{
					'url': resolve_download_url('models-3.0.0', 'peppa_wutz.hash'),
					'path': resolve_relative_path('../.assets/models/peppa_wutz.hash')
				}
			},
			'sources':
			{
				'peppa_wutz':
				{
					'url': resolve_download_url('models-3.0.0', 'peppa_wutz.onnx'),
					'path': resolve_relative_path('../.assets/models/peppa_wutz.onnx')
				}
			},
			'size': (256, 256)
		},
		'fan_68_5':
		{
			'__metadata__':
			{
				'vendor': 'VideoKit',
				'license': 'OpenRAIL-M',
				'year': 2024
			},
			'hashes':
			{
				'fan_68_5':
				{
					'url': resolve_download_url('models-3.0.0', 'fan_68_5.hash'),
					'path': resolve_relative_path('../.assets/models/vk_0a1f.hash')
				}
			},
			'sources':
			{
				'fan_68_5':
				{
					'url': resolve_download_url('models-3.0.0', 'fan_68_5.onnx'),
					'path': resolve_relative_path('../.assets/models/vk_0a1f.onnx')
				}
			}
		}
	}


def get_inference_pool() -> InferencePool:
	landmark_detector_model = state_manager.get_item('landmark_detector_model') or '2dfan4'
	model_names = [ landmark_detector_model, 'fan_68_5' ]
	_, model_source_set = collect_model_downloads()

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	landmark_detector_model = state_manager.get_item('landmark_detector_model') or '2dfan4'
	model_names = [ landmark_detector_model, 'fan_68_5' ]
	inference_manager.clear_inference_pool(__name__, model_names)


def collect_model_downloads() -> Tuple[DownloadSet, DownloadSet]:
	model_set = create_static_model_set('full')
	model_hash_set =\
	{
		'fan_68_5': model_set.get('fan_68_5').get('hashes').get('fan_68_5')
	}
	model_source_set =\
	{
		'fan_68_5': model_set.get('fan_68_5').get('sources').get('fan_68_5')
	}

	for landmark_detector_model in [ '2dfan4', 'peppa_wutz' ]:
		if state_manager.get_item('landmark_detector_model') in [ 'many', landmark_detector_model ]:
			model_hash_set[landmark_detector_model] = model_set.get(landmark_detector_model).get('hashes').get(landmark_detector_model)
			model_source_set[landmark_detector_model] = model_set.get(landmark_detector_model).get('sources').get(landmark_detector_model)

	return model_hash_set, model_source_set


def pre_check() -> bool:
	model_hash_set, model_source_set = collect_model_downloads()

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def detect_region_landmark(vision_frame : VisionFrame, bounding_box : BoundingBox, region_angle : Angle) -> Tuple[RegionLandmark68, Score]:
	region_landmark_2dfan4 = None
	region_landmark_peppa_wutz = None
	region_landmark_score_2dfan4 = 0.0
	region_landmark_score_peppa_wutz = 0.0

	if state_manager.get_item('landmark_detector_model') in [ 'many', '2dfan4' ]:
		region_landmark_2dfan4, region_landmark_score_2dfan4 = detect_with_2dfan4(vision_frame, bounding_box, region_angle)

	if state_manager.get_item('landmark_detector_model') in [ 'many', 'peppa_wutz' ]:
		region_landmark_peppa_wutz, region_landmark_score_peppa_wutz = detect_with_peppa_wutz(vision_frame, bounding_box, region_angle)

	if region_landmark_score_2dfan4 > region_landmark_score_peppa_wutz - 0.2:
		return region_landmark_2dfan4, region_landmark_score_2dfan4
	return region_landmark_peppa_wutz, region_landmark_score_peppa_wutz


def detect_with_2dfan4(temp_vision_frame: VisionFrame, bounding_box: BoundingBox, region_angle: Angle) -> Tuple[RegionLandmark68, Score]:
	model_size = create_static_model_set('full').get('2dfan4').get('size')
	scale = 195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max().clip(1, None)
	translation = (model_size[0] - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
	rotation_matrix, rotation_size = create_rotation_matrix_and_size(region_angle, model_size)
	crop_vision_frame, affine_matrix = warp_region_by_translation(temp_vision_frame, translation, scale, model_size)
	crop_vision_frame = cv2.warpAffine(crop_vision_frame, rotation_matrix, rotation_size)
	crop_vision_frame = conditional_optimize_contrast(crop_vision_frame)
	crop_vision_frame = crop_vision_frame.transpose(2, 0, 1).astype(numpy.float32) / 255.0
	region_landmark_68, region_heatmap = forward_with_2dfan4(crop_vision_frame)
	region_landmark_68 = region_landmark_68[:, :, :2][0] / 64 * 256
	region_landmark_68 = transform_points(region_landmark_68, cv2.invertAffineTransform(rotation_matrix))
	region_landmark_68 = transform_points(region_landmark_68, cv2.invertAffineTransform(affine_matrix))
	region_landmark_score_68 = numpy.amax(region_heatmap, axis = (2, 3))
	region_landmark_score_68 = numpy.mean(region_landmark_score_68)
	region_landmark_score_68 = numpy.interp(region_landmark_score_68, [ 0, 0.9 ], [ 0, 1 ])
	return region_landmark_68, region_landmark_score_68


def detect_with_peppa_wutz(temp_vision_frame : VisionFrame, bounding_box : BoundingBox, region_angle : Angle) -> Tuple[RegionLandmark68, Score]:
	model_size = create_static_model_set('full').get('peppa_wutz').get('size')
	scale = 195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max().clip(1, None)
	translation = (model_size[0] - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
	rotation_matrix, rotation_size = create_rotation_matrix_and_size(region_angle, model_size)
	crop_vision_frame, affine_matrix = warp_region_by_translation(temp_vision_frame, translation, scale, model_size)
	crop_vision_frame = cv2.warpAffine(crop_vision_frame, rotation_matrix, rotation_size)
	crop_vision_frame = conditional_optimize_contrast(crop_vision_frame)
	crop_vision_frame = crop_vision_frame.transpose(2, 0, 1).astype(numpy.float32) / 255.0
	crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis = 0)
	prediction = forward_with_peppa_wutz(crop_vision_frame)
	region_landmark_68 = prediction.reshape(-1, 3)[:, :2] / 64 * model_size[0]
	region_landmark_68 = transform_points(region_landmark_68, cv2.invertAffineTransform(rotation_matrix))
	region_landmark_68 = transform_points(region_landmark_68, cv2.invertAffineTransform(affine_matrix))
	region_landmark_score_68 = prediction.reshape(-1, 3)[:, 2].mean()
	region_landmark_score_68 = numpy.interp(region_landmark_score_68, [ 0, 0.95 ], [ 0, 1 ])
	return region_landmark_68, region_landmark_score_68


def conditional_optimize_contrast(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_RGB2Lab)
	if numpy.mean(crop_vision_frame[:, :, 0]) < 30: #type:ignore[arg-type]
		crop_vision_frame[:, :, 0] = cv2.createCLAHE(clipLimit = 2).apply(crop_vision_frame[:, :, 0])
	crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_Lab2RGB)
	return crop_vision_frame


def estimate_region_landmark_68_5(region_landmark_5 : RegionLandmark5) -> RegionLandmark68:
	affine_matrix = estimate_matrix_by_region_landmark_5(region_landmark_5, 'ffhq_512', (1, 1))
	region_landmark_5 = cv2.transform(region_landmark_5.reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
	region_landmark_68_5 = forward_fan_68_5(region_landmark_5)
	region_landmark_68_5 = cv2.transform(region_landmark_68_5.reshape(1, -1, 2), cv2.invertAffineTransform(affine_matrix)).reshape(-1, 2)
	return region_landmark_68_5


def forward_with_2dfan4(crop_vision_frame : VisionFrame) -> Tuple[Prediction, Prediction]:
	landmark_detector = get_inference_pool().get('2dfan4')

	with conditional_thread_semaphore():
		prediction = landmark_detector.run(None,
		{
			'input': [ crop_vision_frame ]
		})

	return prediction


def forward_with_peppa_wutz(crop_vision_frame : VisionFrame) -> Prediction:
	landmark_detector = get_inference_pool().get('peppa_wutz')

	with conditional_thread_semaphore():
		prediction = landmark_detector.run(None,
		{
			'input': crop_vision_frame
		})[0]

	return prediction


def forward_fan_68_5(region_landmark_5 : RegionLandmark5) -> RegionLandmark68:
	landmark_detector = get_inference_pool().get('fan_68_5')

	with conditional_thread_semaphore():
		region_landmark_68_5 = landmark_detector.run(None,
		{
			'input': [ region_landmark_5 ]
		})[0][0]

	return region_landmark_68_5
