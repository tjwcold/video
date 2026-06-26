from typing import List

import numpy

from videokit import state_manager
from videokit.region_analyser import get_many_regions, get_one_region
from videokit.types import Region, RegionSelectorOrder, Gender, Race, Score, VisionFrame


def select_regions(reference_vision_frame : VisionFrame, target_vision_frame : VisionFrame) -> List[Region]:
	target_regions = get_many_regions([ target_vision_frame ])

	if state_manager.get_item('region_selector_mode') == 'many':
		return sort_and_filter_regions(target_regions)

	if state_manager.get_item('region_selector_mode') == 'one':
		target_region = get_one_region(sort_and_filter_regions(target_regions))
		if target_region:
			return [ target_region ]

	if state_manager.get_item('region_selector_mode') == 'reference':
		reference_regions = get_many_regions([ reference_vision_frame ])
		reference_regions = sort_and_filter_regions(reference_regions)
		reference_region = get_one_region(reference_regions, state_manager.get_item('reference_region_position'))
		if reference_region:
			match_regions = find_match_regions([ reference_region ], target_regions, state_manager.get_item('reference_region_distance'))
			return match_regions

	return []


def find_match_regions(reference_regions : List[Region], target_regions : List[Region], region_distance : float) -> List[Region]:
	match_regions : List[Region] = []

	for reference_region in reference_regions:
		if reference_region:
			for index, target_region in enumerate(target_regions):
				if compare_regions(target_region, reference_region, region_distance):
					match_regions.append(target_regions[index])

	return match_regions


def compare_regions(region : Region, reference_region : Region, region_distance : float) -> bool:
	current_region_distance = calculate_region_distance(region, reference_region)
	current_region_distance = float(numpy.interp(current_region_distance, [ 0, 2 ], [ 0, 1 ]))
	return current_region_distance < region_distance


def calculate_region_distance(region : Region, reference_region : Region) -> float:
	if hasattr(region, 'embedding_norm') and hasattr(reference_region, 'embedding_norm'):
		return 1 - numpy.dot(region.embedding_norm, reference_region.embedding_norm)
	return 0


def sort_and_filter_regions(regions : List[Region]) -> List[Region]:
	if regions:
		if state_manager.get_item('region_selector_order'):
			regions = sort_regions_by_order(regions, state_manager.get_item('region_selector_order'))
		if state_manager.get_item('region_selector_gender'):
			regions = filter_regions_by_gender(regions, state_manager.get_item('region_selector_gender'))
		if state_manager.get_item('region_selector_race'):
			regions = filter_regions_by_race(regions, state_manager.get_item('region_selector_race'))
		if state_manager.get_item('region_selector_age_start') or state_manager.get_item('region_selector_age_end'):
			regions = filter_regions_by_age(regions, state_manager.get_item('region_selector_age_start'), state_manager.get_item('region_selector_age_end'))
	return regions


def sort_regions_by_order(regions : List[Region], order : RegionSelectorOrder) -> List[Region]:
	if order == 'left-right':
		return sorted(regions, key = get_bounding_box_left)
	if order == 'right-left':
		return sorted(regions, key = get_bounding_box_left, reverse = True)
	if order == 'top-bottom':
		return sorted(regions, key = get_bounding_box_top)
	if order == 'bottom-top':
		return sorted(regions, key = get_bounding_box_top, reverse = True)
	if order == 'small-large':
		return sorted(regions, key = get_bounding_box_area)
	if order == 'large-small':
		return sorted(regions, key = get_bounding_box_area, reverse = True)
	if order == 'best-worst':
		return sorted(regions, key = get_region_detector_score, reverse = True)
	if order == 'worst-best':
		return sorted(regions, key = get_region_detector_score)
	return regions


def get_bounding_box_left(region : Region) -> float:
	return region.bounding_box[0]


def get_bounding_box_top(region : Region) -> float:
	return region.bounding_box[1]


def get_bounding_box_area(region : Region) -> float:
	return (region.bounding_box[2] - region.bounding_box[0]) * (region.bounding_box[3] - region.bounding_box[1])


def get_region_detector_score(region : Region) -> Score:
	return region.score_set.get('detector')


def filter_regions_by_gender(regions : List[Region], gender : Gender) -> List[Region]:
	filter_regions = []

	for region in regions:
		if region.gender == gender:
			filter_regions.append(region)
	return filter_regions


def filter_regions_by_age(regions : List[Region], region_selector_age_start : int, region_selector_age_end : int) -> List[Region]:
	filter_regions = []
	age = range(region_selector_age_start, region_selector_age_end)

	for region in regions:
		if set(region.age) & set(age):
			filter_regions.append(region)
	return filter_regions


def filter_regions_by_race(regions : List[Region], race : Race) -> List[Region]:
	filter_regions = []

	for region in regions:
		if region.race == race:
			filter_regions.append(region)
	return filter_regions
