from functools import lru_cache
from typing import List, Sequence, Tuple

import cv2
import numpy

from videokit import inference_manager, state_manager
from videokit.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from videokit.region_helper import create_rotation_matrix_and_size, create_static_anchors, distance_to_bounding_box, distance_to_region_landmark_5, normalize_bounding_box, transform_bounding_box, transform_points
from videokit.filesystem import resolve_relative_path
from videokit.thread_helper import thread_semaphore
from videokit.types import Angle, BoundingBox, Detection, DownloadScope, DownloadSet, RegionLandmark5, InferencePool, Margin, ModelSet, Score, VisionFrame
from videokit.vision import restrict_frame, unpack_resolution


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'retinaface':
		{
			'__metadata__':
			{
				'vendor': 'InsightFace',
				'license': 'Non-Commercial',
				'year': 2020
			},
			'hashes':
			{
				'retinaface':
				{
					'url': resolve_download_url('models-3.0.0', 'retinaface_10g.hash'),
					'path': resolve_relative_path('../.assets/models/vk_1b2e.hash')
				}
			},
			'sources':
			{
				'retinaface':
				{
					'url': resolve_download_url('models-3.0.0', 'retinaface_10g.onnx'),
					'path': resolve_relative_path('../.assets/models/vk_1b2e.onnx')
				}
			}
		},
		'scrfd':
		{
			'__metadata__':
			{
				'vendor': 'InsightFace',
				'license': 'Non-Commercial',
				'year': 2021
			},
			'hashes':
			{
				'scrfd':
				{
					'url': resolve_download_url('models-3.0.0', 'scrfd_2.5g.hash'),
					'path': resolve_relative_path('../.assets/models/scrfd_2.5g.hash')
				}
			},
			'sources':
			{
				'scrfd':
				{
					'url': resolve_download_url('models-3.0.0', 'scrfd_2.5g.onnx'),
					'path': resolve_relative_path('../.assets/models/scrfd_2.5g.onnx')
				}
			}
		},
		'yolo_face':
		{
			'__metadata__':
			{
				'vendor': 'derronqi',
				'license': 'GPL-3.0',
				'year': 2022
			},
			'hashes':
			{
				'yolo_face':
				{
					'url': resolve_download_url('models-3.0.0', 'yoloface_8n.hash'),
					'path': resolve_relative_path('../.assets/models/yoloface_8n.hash')
				}
			},
			'sources':
			{
				'yolo_face':
				{
					'url': resolve_download_url('models-3.0.0', 'yoloface_8n.onnx'),
					'path': resolve_relative_path('../.assets/models/yoloface_8n.onnx')
				}
			}
		},
		'yunet':
		{
			'__metadata__':
			{
				'vendor': 'OpenCV',
				'license': 'MIT',
				'year': 2023
			},
			'hashes':
			{
				'yunet':
				{
					'url': resolve_download_url('models-3.4.0', 'yunet_2023_mar.hash'),
					'path': resolve_relative_path('../.assets/models/yunet_2023_mar.hash')
				}
			},
			'sources':
			{
				'yunet':
				{
					'url': resolve_download_url('models-3.4.0', 'yunet_2023_mar.onnx'),
					'path': resolve_relative_path('../.assets/models/yunet_2023_mar.onnx')
				}
			}
		}
	}


def get_inference_pool() -> InferencePool:
	region_detector_model = state_manager.get_item('region_detector_model') or 'retinaface'
	model_names = [ region_detector_model ]
	_, model_source_set = collect_model_downloads()

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	region_detector_model = state_manager.get_item('region_detector_model') or 'retinaface'
	model_names = [ region_detector_model ]
	inference_manager.clear_inference_pool(__name__, model_names)


def collect_model_downloads() -> Tuple[DownloadSet, DownloadSet]:
	model_set = create_static_model_set('full')
	model_hash_set = {}
	model_source_set = {}

	for region_detector_model in [ 'retinaface', 'scrfd', 'yolo_face', 'yunet' ]:
		if state_manager.get_item('region_detector_model') in [ 'many', region_detector_model ]:
			model_hash_set[region_detector_model] = model_set.get(region_detector_model).get('hashes').get(region_detector_model)
			model_source_set[region_detector_model] = model_set.get(region_detector_model).get('sources').get(region_detector_model)

	return model_hash_set, model_source_set


def pre_check() -> bool:
	model_hash_set, model_source_set = collect_model_downloads()

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def detect_regions(vision_frame : VisionFrame) -> Tuple[List[BoundingBox], List[Score], List[RegionLandmark5]]:
	margin_top, margin_right, margin_bottom, margin_left = prepare_margin(vision_frame)
	margin_vision_frame = numpy.pad(vision_frame, ((margin_top, margin_bottom), (margin_left, margin_right), (0, 0)))
	all_bounding_boxes : List[BoundingBox] = []
	all_region_scores : List[Score] = []
	all_region_landmarks_5 : List[RegionLandmark5] = []

	if state_manager.get_item('region_detector_model') in [ 'many', 'retinaface' ]:
		bounding_boxes, region_scores, region_landmarks_5 = detect_with_retinaface(margin_vision_frame, state_manager.get_item('region_detector_size'))
		all_bounding_boxes.extend(bounding_boxes)
		all_region_scores.extend(region_scores)
		all_region_landmarks_5.extend(region_landmarks_5)

	if state_manager.get_item('region_detector_model') in [ 'many', 'scrfd' ]:
		bounding_boxes, region_scores, region_landmarks_5 = detect_with_scrfd(margin_vision_frame, state_manager.get_item('region_detector_size'))
		all_bounding_boxes.extend(bounding_boxes)
		all_region_scores.extend(region_scores)
		all_region_landmarks_5.extend(region_landmarks_5)

	if state_manager.get_item('region_detector_model') in [ 'many', 'yolo_face' ]:
		bounding_boxes, region_scores, region_landmarks_5 = detect_with_yolo_face(margin_vision_frame, state_manager.get_item('region_detector_size'))
		all_bounding_boxes.extend(bounding_boxes)
		all_region_scores.extend(region_scores)
		all_region_landmarks_5.extend(region_landmarks_5)

	if state_manager.get_item('region_detector_model') == 'yunet':
		bounding_boxes, region_scores, region_landmarks_5 = detect_with_yunet(margin_vision_frame, state_manager.get_item('region_detector_size'))
		all_bounding_boxes.extend(bounding_boxes)
		all_region_scores.extend(region_scores)
		all_region_landmarks_5.extend(region_landmarks_5)

	all_bounding_boxes = [ normalize_bounding_box(all_bounding_box) - numpy.array([ margin_left, margin_top, margin_left, margin_top ]) for all_bounding_box in all_bounding_boxes ]
	all_region_landmarks_5 = [ all_region_landmark_5 - numpy.array([ margin_left, margin_top ]) for all_region_landmark_5 in all_region_landmarks_5 ]
	return all_bounding_boxes, all_region_scores, all_region_landmarks_5


def prepare_margin(vision_frame : VisionFrame) -> Margin:
	margin_top = int(vision_frame.shape[0] * numpy.interp(state_manager.get_item('region_detector_margin')[0], [ 0, 100 ], [ 0, 0.5 ]))
	margin_right = int(vision_frame.shape[1] * numpy.interp(state_manager.get_item('region_detector_margin')[1], [ 0, 100 ], [ 0, 0.5 ]))
	margin_bottom = int(vision_frame.shape[0] * numpy.interp(state_manager.get_item('region_detector_margin')[2], [ 0, 100 ], [ 0, 0.5 ]))
	margin_left = int(vision_frame.shape[1] * numpy.interp(state_manager.get_item('region_detector_margin')[3], [ 0, 100 ], [ 0, 0.5 ]))
	return margin_top, margin_right, margin_bottom, margin_left


def detect_regions_by_angle(vision_frame : VisionFrame, region_angle : Angle) -> Tuple[List[BoundingBox], List[Score], List[RegionLandmark5]]:
	rotation_matrix, rotation_size = create_rotation_matrix_and_size(region_angle, vision_frame.shape[:2][::-1])
	rotation_vision_frame = cv2.warpAffine(vision_frame, rotation_matrix, rotation_size)
	rotation_inverse_matrix = cv2.invertAffineTransform(rotation_matrix)
	bounding_boxes, region_scores, region_landmarks_5 = detect_regions(rotation_vision_frame)
	bounding_boxes = [ transform_bounding_box(bounding_box, rotation_inverse_matrix) for bounding_box in bounding_boxes ]
	region_landmarks_5 = [ transform_points(region_landmark_5, rotation_inverse_matrix) for region_landmark_5 in region_landmarks_5 ]
	return bounding_boxes, region_scores, region_landmarks_5


def detect_with_retinaface(vision_frame : VisionFrame, region_detector_size : str) -> Tuple[List[BoundingBox], List[Score], List[RegionLandmark5]]:
	bounding_boxes = []
	region_scores = []
	region_landmarks_5 = []
	feature_strides = [ 8, 16, 32 ]
	feature_map_channel = 3
	anchor_total = 2
	region_detector_score = state_manager.get_item('region_detector_score')
	region_detector_width, region_detector_height = unpack_resolution(region_detector_size)
	temp_vision_frame = restrict_frame(vision_frame, (region_detector_width, region_detector_height))
	ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
	ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
	detect_vision_frame = prepare_detect_frame(temp_vision_frame, region_detector_size)
	detect_vision_frame = normalize_detect_frame(detect_vision_frame, [ -1, 1 ])
	detection = forward_with_retinaface(detect_vision_frame)

	for index, feature_stride in enumerate(feature_strides):
		region_scores_raw = detection[index]
		keep_indices = numpy.where(region_scores_raw >= region_detector_score)[0]

		if numpy.any(keep_indices):
			stride_height = region_detector_height // feature_stride
			stride_width = region_detector_width // feature_stride
			anchors = create_static_anchors(feature_stride, anchor_total, stride_height, stride_width)
			bounding_boxes_raw = detection[index + feature_map_channel] * feature_stride
			region_landmarks_5_raw = detection[index + feature_map_channel * 2] * feature_stride

			for bounding_box_raw in distance_to_bounding_box(anchors, bounding_boxes_raw)[keep_indices]:
				bounding_boxes.append(numpy.array(
				[
					bounding_box_raw[0] * ratio_width,
					bounding_box_raw[1] * ratio_height,
					bounding_box_raw[2] * ratio_width,
					bounding_box_raw[3] * ratio_height
				]))

			for region_score_raw in region_scores_raw[keep_indices]:
				region_scores.append(region_score_raw[0])

			for region_landmark_raw_5 in distance_to_region_landmark_5(anchors, region_landmarks_5_raw)[keep_indices]:
				region_landmarks_5.append(region_landmark_raw_5 * [ ratio_width, ratio_height ])

	return bounding_boxes, region_scores, region_landmarks_5


def detect_with_scrfd(vision_frame : VisionFrame, region_detector_size : str) -> Tuple[List[BoundingBox], List[Score], List[RegionLandmark5]]:
	bounding_boxes = []
	region_scores = []
	region_landmarks_5 = []
	feature_strides = [ 8, 16, 32 ]
	feature_map_channel = 3
	anchor_total = 2
	region_detector_score = state_manager.get_item('region_detector_score')
	region_detector_width, region_detector_height = unpack_resolution(region_detector_size)
	temp_vision_frame = restrict_frame(vision_frame, (region_detector_width, region_detector_height))
	ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
	ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
	detect_vision_frame = prepare_detect_frame(temp_vision_frame, region_detector_size)
	detect_vision_frame = normalize_detect_frame(detect_vision_frame, [ -1, 1 ])
	detection = forward_with_scrfd(detect_vision_frame)

	for index, feature_stride in enumerate(feature_strides):
		region_scores_raw = detection[index]
		keep_indices = numpy.where(region_scores_raw >= region_detector_score)[0]

		if numpy.any(keep_indices):
			stride_height = region_detector_height // feature_stride
			stride_width = region_detector_width // feature_stride
			anchors = create_static_anchors(feature_stride, anchor_total, stride_height, stride_width)
			bounding_boxes_raw = detection[index + feature_map_channel] * feature_stride
			region_landmarks_5_raw = detection[index + feature_map_channel * 2] * feature_stride

			for bounding_box_raw in distance_to_bounding_box(anchors, bounding_boxes_raw)[keep_indices]:
				bounding_boxes.append(numpy.array(
				[
					bounding_box_raw[0] * ratio_width,
					bounding_box_raw[1] * ratio_height,
					bounding_box_raw[2] * ratio_width,
					bounding_box_raw[3] * ratio_height
				]))

			for region_score_raw in region_scores_raw[keep_indices]:
				region_scores.append(region_score_raw[0])

			for region_landmark_raw_5 in distance_to_region_landmark_5(anchors, region_landmarks_5_raw)[keep_indices]:
				region_landmarks_5.append(region_landmark_raw_5 * [ ratio_width, ratio_height ])

	return bounding_boxes, region_scores, region_landmarks_5


def detect_with_yolo_face(vision_frame : VisionFrame, region_detector_size : str) -> Tuple[List[BoundingBox], List[Score], List[RegionLandmark5]]:
	bounding_boxes = []
	region_scores = []
	region_landmarks_5 = []
	region_detector_score = state_manager.get_item('region_detector_score')
	region_detector_width, region_detector_height = unpack_resolution(region_detector_size)
	temp_vision_frame = restrict_frame(vision_frame, (region_detector_width, region_detector_height))
	ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
	ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
	detect_vision_frame = prepare_detect_frame(temp_vision_frame, region_detector_size)
	detect_vision_frame = normalize_detect_frame(detect_vision_frame, [ 0, 1 ])
	detection = forward_with_yolo_face(detect_vision_frame)
	detection = numpy.squeeze(detection).T
	bounding_boxes_raw, region_scores_raw, region_landmarks_5_raw = numpy.split(detection, [ 4, 5 ], axis = 1)
	keep_indices = numpy.where(region_scores_raw > region_detector_score)[0]

	if numpy.any(keep_indices):
		bounding_boxes_raw, region_scores_raw, region_landmarks_5_raw = bounding_boxes_raw[keep_indices], region_scores_raw[keep_indices], region_landmarks_5_raw[keep_indices]

		for bounding_box_raw in bounding_boxes_raw:
			bounding_boxes.append(numpy.array(
			[
				(bounding_box_raw[0] - bounding_box_raw[2] / 2) * ratio_width,
				(bounding_box_raw[1] - bounding_box_raw[3] / 2) * ratio_height,
				(bounding_box_raw[0] + bounding_box_raw[2] / 2) * ratio_width,
				(bounding_box_raw[1] + bounding_box_raw[3] / 2) * ratio_height
			]))

		region_scores = region_scores_raw.ravel().tolist()
		region_landmarks_5_raw[:, 0::3] = (region_landmarks_5_raw[:, 0::3]) * ratio_width
		region_landmarks_5_raw[:, 1::3] = (region_landmarks_5_raw[:, 1::3]) * ratio_height

		for region_landmark_raw_5 in region_landmarks_5_raw:
			region_landmarks_5.append(numpy.array(region_landmark_raw_5.reshape(-1, 3)[:, :2]))

	return bounding_boxes, region_scores, region_landmarks_5


def detect_with_yunet(vision_frame : VisionFrame, region_detector_size : str) -> Tuple[List[BoundingBox], List[Score], List[RegionLandmark5]]:
	bounding_boxes = []
	region_scores = []
	region_landmarks_5 = []
	feature_strides = [ 8, 16, 32 ]
	feature_map_channel = 3
	anchor_total = 1
	region_detector_score = state_manager.get_item('region_detector_score')
	region_detector_width, region_detector_height = unpack_resolution(region_detector_size)
	temp_vision_frame = restrict_frame(vision_frame, (region_detector_width, region_detector_height))
	ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
	ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
	detect_vision_frame = prepare_detect_frame(temp_vision_frame, region_detector_size)
	detect_vision_frame = normalize_detect_frame(detect_vision_frame, [ 0, 255 ])
	detection = forward_with_yunet(detect_vision_frame)

	for index, feature_stride in enumerate(feature_strides):
		region_scores_raw = (detection[index] * detection[index + feature_map_channel]).reshape(-1)
		keep_indices = numpy.where(region_scores_raw >= region_detector_score)[0]

		if numpy.any(keep_indices):
			stride_height = region_detector_height // feature_stride
			stride_width = region_detector_width // feature_stride
			anchors = create_static_anchors(feature_stride, anchor_total, stride_height, stride_width)
			bounding_boxes_center = detection[index + feature_map_channel * 2].squeeze(0)[:, :2] * feature_stride + anchors
			bounding_boxes_size = numpy.exp(detection[index + feature_map_channel * 2].squeeze(0)[:, 2:4]) * feature_stride
			region_landmarks_5_raw = detection[index + feature_map_channel * 3].squeeze(0)

			bounding_boxes_raw = numpy.stack(
			[
				bounding_boxes_center[:, 0] - bounding_boxes_size[:, 0] / 2,
				bounding_boxes_center[:, 1] - bounding_boxes_size[:, 1] / 2,
				bounding_boxes_center[:, 0] + bounding_boxes_size[:, 0] / 2,
				bounding_boxes_center[:, 1] + bounding_boxes_size[:, 1] / 2
			], axis = -1)

			for bounding_box_raw in bounding_boxes_raw[keep_indices]:
				bounding_boxes.append(numpy.array(
				[
					bounding_box_raw[0] * ratio_width,
					bounding_box_raw[1] * ratio_height,
					bounding_box_raw[2] * ratio_width,
					bounding_box_raw[3] * ratio_height
				]))

			region_scores.extend(region_scores_raw[keep_indices])
			region_landmarks_5_raw = numpy.concatenate(
			[
				region_landmarks_5_raw[:, [ 0, 1 ]] * feature_stride + anchors,
				region_landmarks_5_raw[:, [ 2, 3 ]] * feature_stride + anchors,
				region_landmarks_5_raw[:, [ 4, 5 ]] * feature_stride + anchors,
				region_landmarks_5_raw[:, [ 6, 7 ]] * feature_stride + anchors,
				region_landmarks_5_raw[:, [ 8, 9 ]] * feature_stride + anchors
			], axis = -1).reshape(-1, 5, 2)

			for region_landmark_raw_5 in region_landmarks_5_raw[keep_indices]:
				region_landmarks_5.append(region_landmark_raw_5 * [ ratio_width, ratio_height ])

	return bounding_boxes, region_scores, region_landmarks_5


def forward_with_retinaface(detect_vision_frame : VisionFrame) -> Detection:
	region_detector = get_inference_pool().get('retinaface')

	with thread_semaphore():
		detection = region_detector.run(None,
		{
			'input': detect_vision_frame
		})

	return detection


def forward_with_scrfd(detect_vision_frame : VisionFrame) -> Detection:
	region_detector = get_inference_pool().get('scrfd')

	with thread_semaphore():
		detection = region_detector.run(None,
		{
			'input': detect_vision_frame
		})

	return detection


def forward_with_yolo_face(detect_vision_frame : VisionFrame) -> Detection:
	region_detector = get_inference_pool().get('yolo_face')

	with thread_semaphore():
		detection = region_detector.run(None,
		{
			'input': detect_vision_frame
		})

	return detection


def forward_with_yunet(detect_vision_frame : VisionFrame) -> Detection:
	region_detector = get_inference_pool().get('yunet')

	with thread_semaphore():
		detection = region_detector.run(None,
		{
			'input': detect_vision_frame
		})

	return detection


def prepare_detect_frame(temp_vision_frame : VisionFrame, region_detector_size : str) -> VisionFrame:
	region_detector_width, region_detector_height = unpack_resolution(region_detector_size)
	detect_vision_frame = numpy.zeros((region_detector_height, region_detector_width, 3))
	detect_vision_frame[:temp_vision_frame.shape[0], :temp_vision_frame.shape[1], :] = temp_vision_frame
	detect_vision_frame = numpy.expand_dims(detect_vision_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
	return detect_vision_frame


def normalize_detect_frame(detect_vision_frame : VisionFrame, normalize_range : Sequence[int]) -> VisionFrame:
	if normalize_range == [ -1, 1 ]:
		return (detect_vision_frame - 127.5) / 128.0
	if normalize_range == [ 0, 1 ]:
		return detect_vision_frame / 255.0
	return detect_vision_frame
