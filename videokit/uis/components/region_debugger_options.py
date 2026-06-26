from typing import List, Optional

import gradio

from videokit import state_manager, translator
from videokit.processors.modules.region_debugger import choices as region_debugger_choices
from videokit.processors.modules.region_debugger.types import RegionDebuggerItem
from videokit.uis.core import get_ui_component, register_ui_component

REGION_DEBUGGER_ITEMS_CHECKBOX_GROUP : Optional[gradio.CheckboxGroup] = None


def render() -> None:
	global REGION_DEBUGGER_ITEMS_CHECKBOX_GROUP

	has_region_debugger = 'region_debugger' in state_manager.get_item('processors')
	REGION_DEBUGGER_ITEMS_CHECKBOX_GROUP = gradio.CheckboxGroup(
		label = translator.get('uis.items_checkbox_group', 'videokit.processors.modules.region_debugger'),
		choices = region_debugger_choices.region_debugger_items,
		value = state_manager.get_item('region_debugger_items'),
		visible = has_region_debugger
	)
	register_ui_component('region_debugger_items_checkbox_group', REGION_DEBUGGER_ITEMS_CHECKBOX_GROUP)


def listen() -> None:
	REGION_DEBUGGER_ITEMS_CHECKBOX_GROUP.change(update_region_debugger_items, inputs = REGION_DEBUGGER_ITEMS_CHECKBOX_GROUP)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = REGION_DEBUGGER_ITEMS_CHECKBOX_GROUP)


def remote_update(processors : List[str]) -> gradio.CheckboxGroup:
	has_region_debugger = 'region_debugger' in processors
	return gradio.CheckboxGroup(visible = has_region_debugger)


def update_region_debugger_items(region_debugger_items : List[RegionDebuggerItem]) -> None:
	state_manager.set_item('region_debugger_items', region_debugger_items)
