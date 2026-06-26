import gradio

from videokit import state_manager
from videokit.uis.components import about, age_modifier_options, background_remover_options, common_options, content_blend_options, download, execution, execution_thread_count, expression_restorer_options, region_debugger_options, region_detector, portrait_editor_options, quality_enhancer_options, landmark_detector, region_masker, region_selector, style_transfer_options, frame_colorizer_options, frame_enhancer_options, instant_runner, job_manager, job_runner, lip_syncer_options, memory, output, output_options, preview, preview_options, processors, source, target, temp_frame, terminal, trim_frame, ui_workflow, voice_extractor


def pre_check() -> bool:
	return True


def render() -> gradio.Blocks:
	with gradio.Blocks() as layout:
		with gradio.Row():
			with gradio.Column(scale = 4):
				with gradio.Blocks():
					about.render()
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
					region_debugger_options.render()
			with gradio.Blocks():
				portrait_editor_options.render()
			with gradio.Blocks():
				quality_enhancer_options.render()
			with gradio.Blocks():
				style_transfer_options.render()
				with gradio.Blocks():
					frame_colorizer_options.render()
				with gradio.Blocks():
					frame_enhancer_options.render()
				with gradio.Blocks():
					lip_syncer_options.render()
				with gradio.Blocks():
					voice_extractor.render()
				with gradio.Blocks():
					execution.render()
					execution_thread_count.render()
				with gradio.Blocks():
					download.render()
				with gradio.Blocks():
					memory.render()
				with gradio.Blocks():
					temp_frame.render()
				with gradio.Blocks():
					output_options.render()
			with gradio.Column(scale = 4):
				with gradio.Blocks():
					source.render()
				with gradio.Blocks():
					target.render()
				with gradio.Blocks():
					output.render()
				with gradio.Blocks():
					terminal.render()
				with gradio.Blocks():
					ui_workflow.render()
					instant_runner.render()
					job_runner.render()
					job_manager.render()
			with gradio.Column(scale = 7):
				with gradio.Blocks():
					preview.render()
					preview_options.render()
				with gradio.Blocks():
					trim_frame.render()
				with gradio.Blocks():
					region_selector.render()
				with gradio.Blocks():
					region_masker.render()
				with gradio.Blocks():
					region_detector.render()
				with gradio.Blocks():
					landmark_detector.render()
				with gradio.Blocks():
					common_options.render()
	return layout


def listen() -> None:
	processors.listen()
	age_modifier_options.listen()
	background_remover_options.listen()
	content_blend_options.listen()
	expression_restorer_options.listen()
	region_debugger_options.listen()
	portrait_editor_options.listen()
	quality_enhancer_options.listen()
	style_transfer_options.listen()
	frame_colorizer_options.listen()
	frame_enhancer_options.listen()
	lip_syncer_options.listen()
	execution.listen()
	execution_thread_count.listen()
	download.listen()
	memory.listen()
	temp_frame.listen()
	output_options.listen()
	source.listen()
	target.listen()
	output.listen()
	instant_runner.listen()
	job_runner.listen()
	job_manager.listen()
	terminal.listen()
	preview.listen()
	preview_options.listen()
	trim_frame.listen()
	region_selector.listen()
	region_masker.listen()
	region_detector.listen()
	landmark_detector.listen()
	voice_extractor.listen()
	common_options.listen()


def run(ui : gradio.Blocks) -> None:
	ui.launch(favicon_path = 'videokit.ico', inbrowser = state_manager.get_item('open_browser'))
