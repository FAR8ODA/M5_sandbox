from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'milestone4'

setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(
        exclude=['test']
    ),

    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py'),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='Team B4',
    maintainer_email='team@example.com',

    description='CPEN 391 Milestone 4: LiDAR-camera fusion, disparity extender, path-aware AEB',

    license='MIT',

    tests_require=['pytest'],

    entry_points={
        'console_scripts': [
            'camera_scan_node = milestone4.camera_scan_node:main',
            'depth_scan_node = milestone4.depth_scan_node:main',
            'fusion_node = milestone4.fusion_node:main',
            'gap_follow_node = milestone4.gap_follow_node:main',
            'safety_node = milestone4.safety_node:main',
            'lap_counter_node = milestone4.lap_counter_node:main',
        ],
    },
)
