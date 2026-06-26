import subprocess

import pytest

from videokit import region_classifier, region_detector, landmark_detector, region_recognizer, state_manager
from videokit.download import conditional_download
from videokit.region_analyser import get_many_regions
from videokit.region_store import clear_static_regions
from videokit.vision import read_static_image
from .helper import get_test_example_file, get_test_examples_directory


@pytest.fixture(scope = 'module', autouse = True)
def before_all() -> None:
	conditional_download(get_test_examples_directory(),
	[
		'https://github.com/VideoKit/VideoKit-assets/releases/download/examples-3.0.0/source.jpg'
	])
	subprocess.run([ 'ffmpeg', '-i', get_test_example_file('source.jpg'), '-vf', 'crop=iw*0.8:ih*0.8', get_test_example_file('source-80crop.jpg') ])
	subprocess.run([ 'ffmpeg', '-i', get_test_example_file('source.jpg'), '-vf', 'crop=iw*0.7:ih*0.7', get_test_example_file('source-70crop.jpg') ])
	subprocess.run([ 'ffmpeg', '-i', get_test_example_file('source.jpg'), '-vf', 'crop=iw*0.6:ih*0.6', get_test_example_file('source-60crop.jpg') ])
	state_manager.init_item('execution_device_ids', [ 0 ])
	state_manager.init_item('execution_providers', [ 'cpu' ])
	state_manager.init_item('download_providers', [ 'github' ])
	state_manager.init_item('region_detector_angles', [ 0 ])
	state_manager.init_item('region_detector_model', 'many')
	state_manager.init_item('region_detector_score', 0.5)
	state_manager.init_item('landmark_detector_model', 'many')
	state_manager.init_item('landmark_detector_score', 0.5)
	region_classifier.pre_check()
	landmark_detector.pre_check()
	region_recognizer.pre_check()


@pytest.fixture(autouse = True)
def before_each() -> None:
	region_classifier.clear_inference_pool()
	region_detector.clear_inference_pool()
	landmark_detector.clear_inference_pool()
	region_recognizer.clear_inference_pool()
	clear_static_regions()


def test_get_one_region_with_retinaface() -> None:
	state_manager.init_item('region_detector_model', 'retinaface')
	state_manager.init_item('region_detector_size', '320x320')
	state_manager.init_item('region_detector_margin', (0, 0, 0, 0))
	region_detector.pre_check()

	source_paths =\
	[
		get_test_example_file('source.jpg'),
		get_test_example_file('source-80crop.jpg'),
		get_test_example_file('source-70crop.jpg'),
		get_test_example_file('source-60crop.jpg')
	]

	for source_path in source_paths:
		source_frame = read_static_image(source_path)
		many_regions = get_many_regions([ source_frame ])

		assert len(many_regions) == 1


def test_get_one_region_with_scrfd() -> None:
	state_manager.init_item('region_detector_model', 'scrfd')
	state_manager.init_item('region_detector_size', '320x320')
	state_manager.init_item('region_detector_margin', (0, 0, 0, 0))
	region_detector.pre_check()

	source_paths =\
	[
		get_test_example_file('source.jpg'),
		get_test_example_file('source-80crop.jpg'),
		get_test_example_file('source-70crop.jpg'),
		get_test_example_file('source-60crop.jpg')
	]

	for source_path in source_paths:
		source_frame = read_static_image(source_path)
		many_regions = get_many_regions([ source_frame ])

		assert len(many_regions) == 1


def test_get_one_region_with_yoloface() -> None:
	state_manager.init_item('region_detector_model', 'yolo_face')
	state_manager.init_item('region_detector_size', '640x640')
	state_manager.init_item('region_detector_margin', (0, 0, 0, 0))
	region_detector.pre_check()

	source_paths =\
	[
		get_test_example_file('source.jpg'),
		get_test_example_file('source-80crop.jpg'),
		get_test_example_file('source-70crop.jpg'),
		get_test_example_file('source-60crop.jpg')
	]

	for source_path in source_paths:
		source_frame = read_static_image(source_path)
		many_regions = get_many_regions([ source_frame ])

		assert len(many_regions) == 1


def test_get_one_region_with_yunet() -> None:
	state_manager.init_item('region_detector_model', 'yunet')
	state_manager.init_item('region_detector_size', '640x640')
	state_manager.init_item('region_detector_margin', (0, 0, 0, 0))
	region_detector.pre_check()

	source_paths =\
	[
		get_test_example_file('source.jpg'),
		get_test_example_file('source-80crop.jpg'),
		get_test_example_file('source-70crop.jpg'),
		get_test_example_file('source-60crop.jpg')
	]

	for source_path in source_paths:
		source_frame = read_static_image(source_path)
		many_regions = get_many_regions([ source_frame ])

		assert len(many_regions) == 1


def test_get_many_regions() -> None:
	source_path = get_test_example_file('source.jpg')
	source_frame = read_static_image(source_path)
	many_regions = get_many_regions([ source_frame, source_frame, source_frame ])

	assert len(many_regions) == 3
