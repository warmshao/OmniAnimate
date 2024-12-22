# -*- coding: utf-8 -*-
# @Time    : 2024/8/16 19:51
# @Project : AnimateMaster
# @FileName: setup.py

from setuptools import setup

setup(
    name='omni_animate',
    version='1.2.0',
    description='OmniAnimate',
    author='warmshao',
    author_email='wenshaoguo1026@gmail.com',
    url='https://github.com/warmshao/OmniAnimate',
    include_package_data=True,
    packages=[
        'omni_animate',
        'omni_animate.models',
        'omni_animate.pipelines',
        'omni_animate.common',
        'omni_animate.trt_models'
    ],
    install_requires=[],
    data_files=[]
)
