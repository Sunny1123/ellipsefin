try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

config = {
	 'description': 'My Project',
	 'author': 'Subhasish Basak',
	 'url': 'URL to get it at.',
	 'download_url': 'Where to download it.',
	 'author_email': 'subhasish.sunny@gmail.com',
	 'version': '0.1',
	 'install_requires': ['pytest','numpy','matplotlib','pillow', 'pyqt5','joblib','tqdm_joblib'],
	 'packages': ['denoise'],
	 'scripts': [],
	 'name': 'Edge Preserving denoising'
	}

setup(**config)
