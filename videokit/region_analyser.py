from typing import List, Optional

import numpy

from videokit import state_manager
from videokit.common_helper import get_first
from videokit.region_classifier import classify_region
from videokit.region_detector import detect_regions, detect_regions_by_angle
from videokit.region_helper import apply_nms, convert_to_region_landmark_5, estimate_region_angle, get_nms_threshold
from videokit.landmark_detector import detect_region_landmark, estimate_region_landmark_68_5
from videokit.region_recognizer import calculate_region_embedding
from videokit.region_store import get_static_regions, set_static_regions
from videokit.types import BoundingBox, Region, RegionLandmark5, RegionLandmarkSet, RegionScoreSet, Score, VisionFrame


def create_regions(vision_frame : VisionFrame, bounding_boxes : List[BoundingBox], region_scores : List[Score], region_landmarks_5 : List[RegionLandmark5]) -> List[Region]:
	regions = []
	nms_threshold = get_nms_threshold(state_manager.get_item('region_detector_model'), state_manager.get_item('region_detector_angles'))
	keep_indices = apply_nms(bounding_boxes, region_scores, state_manager.get_item('region_detector_score'), nms_threshold)

	for index in keep_indices:
		bounding_box = bounding_boxes[index]
		region_score = region_scores[index]
		region_landmark_5 = region_landmarks_5[index]
		region_landmark_5_68 = region_landmark_5
		region_landmark_68_5 = estimate_region_landmark_68_5(region_landmark_5_68)
		region_landmark_68 = region_landmark_68_5
		region_landmark_score_68 = 0.0
		region_angle = estimate_region_angle(region_landmark_68_5)

		if state_manager.get_item('landmark_detector_score') or 0 > 0:
			region_landmark_68, region_landmark_score_68 = detect_region_landmark(vision_frame, bounding_box, region_angle)
		if region_landmark_score_68 > (state_manager.get_item('landmark_detector_score') or 0):
			region_landmark_5_68 = convert_to_region_landmark_5(region_landmark_68)

		region_landmark_set : RegionLandmarkSet =\
		{
			'5': region_landmark_5,
			'5/68': region_landmark_5_68,
			'68': region_landmark_68,
			'68/5': region_landmark_68_5
		}
		region_score_set : RegionScoreSet =\
		{
			'detector': region_score,
			'landmarker': region_landmark_score_68
		}
		region_embedding, region_embedding_norm = calculate_region_embedding(vision_frame, region_landmark_set.get('5/68'))
		gender, age, race = classify_region(vision_frame, region_landmark_set.get('5/68'))
		regions.append(Region(
			bounding_box = bounding_box,
			score_set = region_score_set,
			landmark_set = region_landmark_set,
			angle = region_angle,
			embedding = region_embedding,
			embedding_norm = region_embedding_norm,
			gender = gender,
			age = age,
			race = race
		))
	return regions


def get_one_region(regions : List[Region], position : int = 0) -> Optional[Region]:
	if regions:
		position = min(position, len(regions) - 1)
		return regions[position]
	return None


def get_average_region(regions : List[Region]) -> Optional[Region]:
	region_embeddings = []
	region_embeddings_norm = []

	if regions:
		first_region = get_first(regions)

		for region in regions:
			region_embeddings.append(region.embedding)
			region_embeddings_norm.append(region.embedding_norm)

		return Region(
			bounding_box = first_region.bounding_box,
			score_set = first_region.score_set,
			landmark_set = first_region.landmark_set,
			angle = first_region.angle,
			embedding = numpy.mean(region_embeddings, axis = 0),
			embedding_norm = numpy.mean(region_embeddings_norm, axis = 0),
			gender = first_region.gender,
			age = first_region.age,
			race = first_region.race
		)
	return None


def get_many_regions(vision_frames : List[VisionFrame]) -> List[Region]:
	many_regions : List[Region] = []

	for vision_frame in vision_frames:
		if numpy.any(vision_frame):
			static_regions = get_static_regions(vision_frame)
			if static_regions:
				many_regions.extend(static_regions)
			else:
				all_bounding_boxes = []
				all_region_scores = []
				all_region_landmarks_5 = []

				for region_detector_angle in state_manager.get_item('region_detector_angles'):
					if region_detector_angle == 0:
						bounding_boxes, region_scores, region_landmarks_5 = detect_regions(vision_frame)
					else:
						bounding_boxes, region_scores, region_landmarks_5 = detect_regions_by_angle(vision_frame, region_detector_angle)
					all_bounding_boxes.extend(bounding_boxes)
					all_region_scores.extend(region_scores)
					all_region_landmarks_5.extend(region_landmarks_5)

				if all_bounding_boxes and all_region_scores and all_region_landmarks_5 and (state_manager.get_item('region_detector_score') or 0) > 0:
					regions = create_regions(vision_frame, all_bounding_boxes, all_region_scores, all_region_landmarks_5)

					if regions:
						many_regions.extend(regions)
						set_static_regions(vision_frame, regions)
	return many_regions


def scale_region(target_region : Region, target_vision_frame : VisionFrame, temp_vision_frame : VisionFrame) -> Region:
	scale_x = temp_vision_frame.shape[1] / target_vision_frame.shape[1]
	scale_y = temp_vision_frame.shape[0] / target_vision_frame.shape[0]

	bounding_box = target_region.bounding_box * [ scale_x, scale_y, scale_x, scale_y ]
	landmark_set =\
	{
		'5': target_region.landmark_set.get('5') * numpy.array([ scale_x, scale_y ]),
		'5/68': target_region.landmark_set.get('5/68') * numpy.array([ scale_x, scale_y ]),
		'68': target_region.landmark_set.get('68') * numpy.array([ scale_x, scale_y ]),
		'68/5': target_region.landmark_set.get('68/5') * numpy.array([ scale_x, scale_y ])
	}

	return target_region._replace(
		bounding_box = bounding_box,
		landmark_set = landmark_set
	)
