from collections import namedtuple
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeAlias, TypedDict

import cv2
import numpy
from numpy.typing import NDArray
from onnxruntime import InferenceSession

Scale : TypeAlias = float
Score : TypeAlias = float
Angle : TypeAlias = int

Detection : TypeAlias = NDArray[Any]
Prediction : TypeAlias = NDArray[Any]

BoundingBox : TypeAlias = NDArray[Any]
RegionLandmark5 : TypeAlias = NDArray[Any]
RegionLandmark68 : TypeAlias = NDArray[Any]
RegionLandmarkSet = TypedDict('RegionLandmarkSet',
{
	'5' : RegionLandmark5, #type:ignore[valid-type]
	'5/68' : RegionLandmark5, #type:ignore[valid-type]
	'68' : RegionLandmark68, #type:ignore[valid-type]
	'68/5' : RegionLandmark68 #type:ignore[valid-type]
})
RegionScoreSet = TypedDict('RegionScoreSet',
{
	'detector' : Score,
	'landmarker' : Score
})
Embedding : TypeAlias = NDArray[numpy.float64]
Gender = Literal['female', 'male']
Age : TypeAlias = range
Race = Literal['white', 'black', 'latino', 'asian', 'indian', 'arabic']
Region = namedtuple('Region',
[
	'bounding_box',
	'score_set',
	'landmark_set',
	'angle',
	'embedding',
	'embedding_norm',
	'gender',
	'age',
	'race'
])
RegionSet : TypeAlias = Dict[str, List[Region]]
RegionStore = TypedDict('RegionStore',
{
	'static_regions' : RegionSet
})

Language = Literal['en']
Locales : TypeAlias = Dict[Language, Dict[str, Any]]
LocalePoolSet : TypeAlias = Dict[str, Locales]

VideoCaptureSet : TypeAlias = Dict[str, cv2.VideoCapture]
VideoWriterSet : TypeAlias = Dict[str, cv2.VideoWriter]
CameraCaptureSet : TypeAlias = Dict[str, cv2.VideoCapture]
VideoPoolSet = TypedDict('VideoPoolSet',
{
	'capture': VideoCaptureSet,
	'writer': VideoWriterSet
})
CameraPoolSet = TypedDict('CameraPoolSet',
{
	'capture': CameraCaptureSet
})

ColorMode = Literal['rgb', 'rgba']
VisionFrame : TypeAlias = NDArray[Any]
Mask : TypeAlias = NDArray[Any]
Points : TypeAlias = NDArray[Any]
Distance : TypeAlias = NDArray[Any]
Matrix : TypeAlias = NDArray[Any]
Anchors : TypeAlias = NDArray[Any]
Translation : TypeAlias = NDArray[Any]

AudioBuffer : TypeAlias = bytes
Audio : TypeAlias = NDArray[Any]
AudioChunk : TypeAlias = NDArray[Any]
AudioFrame : TypeAlias = NDArray[Any]
Spectrogram : TypeAlias = NDArray[Any]
Mel : TypeAlias = NDArray[Any]
MelFilterBank : TypeAlias = NDArray[Any]
Voice : TypeAlias = NDArray[Any]
VoiceChunk : TypeAlias = NDArray[Any]

Fps : TypeAlias = float
Duration : TypeAlias = float
Color : TypeAlias = Tuple[int, int, int, int]
Padding : TypeAlias = Tuple[int, int, int, int]
Margin : TypeAlias = Tuple[int, int, int, int]
Orientation = Literal['landscape', 'portrait']
Resolution : TypeAlias = Tuple[int, int]

ProcessState = Literal['checking', 'processing', 'stopping', 'pending']
Args : TypeAlias = Dict[str, Any]
UpdateProgress : TypeAlias = Callable[[int], None]
ProcessStep : TypeAlias = Callable[[str, int, Args], bool]

Content : TypeAlias = Dict[str, Any]

Command : TypeAlias = str
CommandSet : TypeAlias = Dict[str, List[Command]]

WarpTemplate = Literal['arcface_112_v1', 'arcface_112_v2', 'arcface_128', 'dfl_whole_face', 'ffhq_512', 'mtcnn_512', 'styleganex_384']
WarpTemplateSet : TypeAlias = Dict[WarpTemplate, NDArray[Any]]
ProcessMode = Literal['output', 'preview', 'stream']

ErrorCode = Literal[0, 1, 2, 3, 4]
LogLevel = Literal['error', 'warn', 'info', 'debug']
LogLevelSet : TypeAlias = Dict[LogLevel, int]

TableHeader : TypeAlias = str
TableContent : TypeAlias = Any

RegionDetectorModel = Literal['many', 'retinaface', 'scrfd', 'yolo_face', 'yunet']
RegionLandmarkerModel = Literal['many', '2dfan4', 'peppa_wutz']
RegionDetectorSet : TypeAlias = Dict[RegionDetectorModel, List[str]]
RegionSelectorMode = Literal['many', 'one', 'reference']
RegionSelectorOrder = Literal['left-right', 'right-left', 'top-bottom', 'bottom-top', 'small-large', 'large-small', 'best-worst', 'worst-best']
RegionOccluderModel = Literal['many', 'xseg_1', 'xseg_2', 'xseg_3']
RegionParserModel = Literal['bisenet_resnet_18', 'bisenet_resnet_34']
RegionMaskType = Literal['box', 'occlusion', 'area', 'region']
RegionMaskArea = Literal['upper-region', 'lower-region', 'mouth']
RegionMaskRegion = Literal['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip']
RegionMaskRegionSet : TypeAlias = Dict[RegionMaskRegion, int]
RegionMaskAreaSet : TypeAlias = Dict[RegionMaskArea, List[int]]

VoiceExtractorModel = Literal['kim_vocal_1', 'kim_vocal_2', 'uvr_mdxnet']

AudioFormat = Literal['flac', 'm4a', 'mp3', 'ogg', 'opus', 'wav']
ImageFormat = Literal['bmp', 'jpeg', 'png', 'tiff', 'webp']
VideoFormat = Literal['avi', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mxf', 'webm', 'wmv']
TempFrameFormat = Literal['bmp', 'jpeg', 'png', 'tiff']
AudioTypeSet : TypeAlias = Dict[AudioFormat, str]
ImageTypeSet : TypeAlias = Dict[ImageFormat, str]
VideoTypeSet : TypeAlias = Dict[VideoFormat, str]

AudioEncoder = Literal['flac', 'aac', 'libmp3lame', 'libopus', 'libvorbis', 'pcm_s16le', 'pcm_s32le']
VideoEncoder = Literal['libx264', 'libx264rgb', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc', 'h264_amf', 'hevc_amf', 'h264_qsv', 'hevc_qsv', 'h264_videotoolbox', 'hevc_videotoolbox', 'rawvideo']
EncoderSet = TypedDict('EncoderSet',
{
	'audio' : List[AudioEncoder],
	'video' : List[VideoEncoder]
})
VideoPreset = Literal['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']

BenchmarkMode = Literal['warm', 'cold']
BenchmarkResolution = Literal['240p', '360p', '540p', '720p', '1080p', '1440p', '2160p']
BenchmarkSet : TypeAlias = Dict[BenchmarkResolution, str]
BenchmarkCycleSet = TypedDict('BenchmarkCycleSet',
{
	'target_path' : str,
	'cycle_count' : int,
	'average_run' : float,
	'fastest_run' : float,
	'slowest_run' : float,
	'relative_fps' : float
})

WebcamMode = Literal['inline', 'udp', 'v4l2']
StreamMode = Literal['udp', 'v4l2']

ModelOptions : TypeAlias = Dict[str, Any]
ModelSet : TypeAlias = Dict[str, ModelOptions]
ModelInitializer : TypeAlias = NDArray[Any]

ExecutionProvider = Literal['cuda', 'tensorrt', 'rocm', 'migraphx', 'coreml', 'openvino', 'qnn', 'directml', 'cpu']
ExecutionProviderValue = Literal['CPUExecutionProvider', 'CoreMLExecutionProvider', 'CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'MIGraphXExecutionProvider', 'QNNExecutionProvider', 'ROCMExecutionProvider', 'TensorrtExecutionProvider']
ExecutionProviderSet : TypeAlias = Dict[ExecutionProvider, ExecutionProviderValue]
InferenceProvider : TypeAlias = Any
InferenceOptionSet : TypeAlias = Dict[str, Any]
ValueAndUnit = TypedDict('ValueAndUnit',
{
	'value' : int,
	'unit' : str
})
ExecutionDeviceFramework = TypedDict('ExecutionDeviceFramework',
{
	'name' : str,
	'version' : str
})
ExecutionDeviceProduct = TypedDict('ExecutionDeviceProduct',
{
	'vendor' : str,
	'name' : str
})
ExecutionDeviceVideoMemory = TypedDict('ExecutionDeviceVideoMemory',
{
	'total' : Optional[ValueAndUnit],
	'free' : Optional[ValueAndUnit]
})
ExecutionDeviceTemperature = TypedDict('ExecutionDeviceTemperature',
{
	'gpu' : Optional[ValueAndUnit],
	'memory' : Optional[ValueAndUnit]
})
ExecutionDeviceUtilization = TypedDict('ExecutionDeviceUtilization',
{
	'gpu' : Optional[ValueAndUnit],
	'memory' : Optional[ValueAndUnit]
})
ExecutionDevice = TypedDict('ExecutionDevice',
{
	'driver_version' : str,
	'framework' : ExecutionDeviceFramework,
	'product' : ExecutionDeviceProduct,
	'video_memory' : ExecutionDeviceVideoMemory,
	'temperature' : ExecutionDeviceTemperature,
	'utilization' : ExecutionDeviceUtilization
})

DownloadProvider = Literal['github', 'huggingface']
DownloadProviderValue = TypedDict('DownloadProviderValue',
{
	'urls' : List[str],
	'path' : str
})
DownloadProviderSet : TypeAlias = Dict[DownloadProvider, DownloadProviderValue]
DownloadScope = Literal['lite', 'full']
Download = TypedDict('Download',
{
	'url' : str,
	'path' : str
})
DownloadSet : TypeAlias = Dict[str, Download]

VideoMemoryStrategy = Literal['strict', 'moderate', 'tolerant']
AppContext = Literal['cli', 'ui']

InferencePool : TypeAlias = Dict[str, InferenceSession]
InferencePoolSet : TypeAlias = Dict[AppContext, Dict[str, InferencePool]]

UiWorkflow = Literal['instant_runner', 'job_runner', 'job_manager']

JobStore = TypedDict('JobStore',
{
	'job_keys' : List[str],
	'step_keys' : List[str]
})
JobOutputSet : TypeAlias = Dict[str, List[str]]
JobStatus = Literal['drafted', 'queued', 'completed', 'failed']
JobStepStatus = Literal['drafted', 'queued', 'started', 'completed', 'failed']
JobStep = TypedDict('JobStep',
{
	'args' : Args,
	'status' : JobStepStatus
})
Job = TypedDict('Job',
{
	'version' : str,
	'date_created' : str,
	'date_updated' : Optional[str],
	'steps' : List[JobStep]
})
JobSet : TypeAlias = Dict[str, Job]

StateKey = Literal\
[
	'command',
	'config_path',
	'temp_path',
	'jobs_path',
	'source_paths',
	'target_path',
	'output_path',
	'source_pattern',
	'target_pattern',
	'output_pattern',
	'download_providers',
	'download_scope',
	'benchmark_mode',
	'benchmark_resolutions',
	'benchmark_cycle_count',
	'region_detector_model',
	'region_detector_size',
	'region_detector_margin',
	'region_detector_angles',
	'region_detector_score',
	'landmark_detector_model',
	'landmark_detector_score',
	'region_selector_mode',
	'region_selector_order',
	'region_selector_gender',
	'region_selector_race',
	'region_selector_age_start',
	'region_selector_age_end',
	'reference_region_position',
	'reference_region_distance',
	'reference_frame_number',
	'region_occluder_model',
	'region_parser_model',
	'region_mask_types',
	'region_mask_areas',
	'region_mask_regions',
	'region_mask_blur',
	'region_mask_padding',
	'voice_extractor_model',
	'trim_frame_start',
	'trim_frame_end',
	'temp_frame_format',
	'keep_temp',
	'output_image_quality',
	'output_image_scale',
	'output_audio_encoder',
	'output_audio_quality',
	'output_audio_volume',
	'output_video_encoder',
	'output_video_preset',
	'output_video_quality',
	'output_video_scale',
	'output_video_fps',
	'processors',
	'open_browser',
	'ui_layouts',
	'ui_workflow',
	'execution_device_ids',
	'execution_providers',
	'execution_thread_count',
	'video_memory_strategy',
	'system_memory_limit',
	'log_level',
	'halt_on_error',
	'job_id',
	'job_status',
	'step_index'
]
State = TypedDict('State',
{
	'command' : str,
	'config_path' : str,
	'temp_path' : str,
	'jobs_path' : str,
	'source_paths' : List[str],
	'target_path' : str,
	'output_path' : str,
	'source_pattern' : str,
	'target_pattern' : str,
	'output_pattern' : str,
	'download_providers' : List[DownloadProvider],
	'download_scope' : DownloadScope,
	'benchmark_mode' : BenchmarkMode,
	'benchmark_resolutions' : List[BenchmarkResolution],
	'benchmark_cycle_count' : int,
	'region_detector_model' : RegionDetectorModel,
	'region_detector_size' : str,
	'region_detector_margin': Margin,
	'region_detector_angles' : List[Angle],
	'region_detector_score' : Score,
	'landmark_detector_model' : RegionLandmarkerModel,
	'landmark_detector_score' : Score,
	'region_selector_mode' : RegionSelectorMode,
	'region_selector_order' : RegionSelectorOrder,
	'region_selector_race' : Race,
	'region_selector_gender' : Gender,
	'region_selector_age_start' : int,
	'region_selector_age_end' : int,
	'reference_region_position' : int,
	'reference_region_distance' : float,
	'reference_frame_number' : int,
	'region_occluder_model' : RegionOccluderModel,
	'region_parser_model' : RegionParserModel,
	'region_mask_types' : List[RegionMaskType],
	'region_mask_areas' : List[RegionMaskArea],
	'region_mask_regions' : List[RegionMaskRegion],
	'region_mask_blur' : float,
	'region_mask_padding' : Padding,
	'voice_extractor_model': VoiceExtractorModel,
	'trim_frame_start' : int,
	'trim_frame_end' : int,
	'temp_frame_format' : TempFrameFormat,
	'keep_temp' : bool,
	'output_image_quality' : int,
	'output_image_scale' : Scale,
	'output_audio_encoder' : AudioEncoder,
	'output_audio_quality' : int,
	'output_audio_volume' : int,
	'output_video_encoder' : VideoEncoder,
	'output_video_preset' : VideoPreset,
	'output_video_quality' : int,
	'output_video_scale' : Scale,
	'output_video_fps' : float,
	'processors' : List[str],
	'open_browser' : bool,
	'ui_layouts' : List[str],
	'ui_workflow' : UiWorkflow,
	'execution_device_ids' : List[int],
	'execution_providers' : List[ExecutionProvider],
	'execution_thread_count' : int,
	'video_memory_strategy' : VideoMemoryStrategy,
	'system_memory_limit' : int,
	'log_level' : LogLevel,
	'halt_on_error' : bool,
	'job_id' : str,
	'job_status' : JobStatus,
	'step_index' : int
})
ApplyStateItem : TypeAlias = Callable[[Any, Any], None]
StateSet : TypeAlias = Dict[AppContext, State]

