from typing import Optional

METADATA =\
{
	'name': 'VideoKit',
	'description': 'Professional video processing and enhancement toolkit',
	'version': '3.6.1',
	'license': 'OpenRAIL-AS',
	'author': 'Henry Ruhs',
	'url': 'https://videokit.io'
}


def get(key : str) -> Optional[str]:
	return METADATA.get(key)
