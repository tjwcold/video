from typing import List, get_args

from videokit.processors.modules.region_debugger.types import RegionDebuggerItem

region_debugger_items : List[RegionDebuggerItem] = list(get_args(RegionDebuggerItem))
