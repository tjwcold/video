from typing import List, Optional

from videokit.hash_helper import create_hash
from videokit.types import Region, RegionStore, VisionFrame

region_store : RegionStore =\
{
	'static_regions': {}
}


def get_region_store() -> RegionStore:
	return region_store


def get_static_regions(vision_frame : VisionFrame) -> Optional[List[Region]]:
	vision_hash = create_hash(vision_frame.tobytes())
	return region_store.get('static_regions').get(vision_hash)


def set_static_regions(vision_frame : VisionFrame, regions : List[Region]) -> None:
	vision_hash = create_hash(vision_frame.tobytes())
	if vision_hash:
		region_store['static_regions'][vision_hash] = regions


def clear_static_regions() -> None:
	region_store['static_regions'].clear()
