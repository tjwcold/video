import gradio

from videokit import benchmarker, state_manager
from videokit.uis.components import about, age_modifier_options, background_remover_options, benchmark, benchmark_options, content_blend_options, download, execution, execution_thread_count, expression_restorer_options, region_debugger_options, portrait_editor_options, quality_enhancer_options, style_transfer_options, frame_colorizer_options, frame_enhancer_options, lip_syncer_options, memory, processors


def pre_check() -> bool:
	return benchmarker.pre_check()


def render() -> gradio.Blocks:
	with gradio.Blocks() as layout:
		with gradio.Row():
			with gradio.Column(scale = 4):
				with gradio.Blocks():
					about.render()
				with gradio.Blocks():
					benchmark_options.render()
				with gradio.Blocks():
					processors.render()
				with gradio.Blocks():
					age_modifier_options.render()
				with gradio.Blocks():
					background_remover_options.render()
				with gradio.Blocks():
					content_blend_options.render()
				with gradio.Blocks():
					expression_restorer_options.render()
				with gradio.Blocks():
					face_debugger_options.render()
				with gradio.Blocks():
					face_editor_options.render()
				with gradio.Blocks():
					face_enhancer_options.render()
				with gradio.Blocks():
					style_transfer_options.render()
				with gradio.Blocks():
					frame_colorizer_options.render()
				with gradio.Blocks():
					frame_enhancer_options.render()
				with gradio.Blocks():
					lip_syncer_options.render()
				with gradio.Blocks():
					execution.render()
					execution_thread_count.render()
				with gradio.Blocks():
					download.render()
				with gradio.Blocks():
					state_manager.set_item('video_memory_strategy', 'tolerant')
					memory.render()
			with gradio.Column(scale = 11):
				with gradio.Blocks():
					benchmark.render()
	return layout


def listen() -> None:
	processors.listen()
	age_modifier_options.listen()
	background_remover_options.listen()
	content_blend_options.listen()
	expression_restorer_options.listen()
	download.listen()
	region_debugger_options.listen()
	portrait_editor_options.listen()
	quality_enhancer_options.listen()
	style_transfer_options.listen()
	frame_colorizer_options.listen()
	frame_enhancer_options.listen()
	lip_syncer_options.listen()
	execution.listen()
	execution_thread_count.listen()
	memory.listen()
	benchmark.listen()
	benchmark_options.listen()


def run(ui : gradio.Blocks) -> None:
	ui.launch(favicon_path = 'videokit.ico', inbrowser = state_manager.get_item('open_browser'))
