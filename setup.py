# coding: utf-8
# Setup script for uth_bert_wrapper


from setuptools import setup, find_packages
import pathlib


here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

required_packages = list(filter(
    lambda x:not (x == '' or x.startswith('#')),
    [_.strip() for _ in (here / 'requirements.txt').read_text().split('\n')]
))


setup(
    name='eval_vl_glue',  # Required
    version='1.0.0',  # Required
    description='eval_vl_glue dev',
    long_description=long_description,
    long_description_content_type='text/markdown',
    #url='https://github.com/pypa/sampleproject',  # Optional
    author='T. Iki',
    author_email='iki@nii.ac.jp',  # Optional
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
    ],
    keywords='lannguage model, transformers, vision-and-language',  # Optional
    packages=[
        'eval_vl_glue', 
        'eval_vl_glue.extractor', 
        'eval_vl_glue.transformers_volta'
    ], # Required
    package_dir={'':'.'}, # Optional
    
    python_requires='>=3.6, <4',
    
    install_requires=required_packages,  # Optional

    #package_data={  # Optional
    #    'sample': ['package_data.dat'],
    #},
    #
    # data_files=[('my_data', ['data/data_file'])],  # Optional
    #
    #entry_points={  # Optional
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    #},
    project_urls={  # Optional
        'Source': 'https://github.com/Alab-NII/eval_vl_glue',
    },
)
