import os
import glob
from setuptools import setup

package_name = 'visual_servoing'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name + '.computer_vision'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/visual_servoing/launch',
            glob.glob(os.path.join('launch', '*launch.xml'))),
        ('share/visual_servoing/launch',
            glob.glob(os.path.join('launch', '*launch.py')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jospeh',
    maintainer_email='jrales@mit.edu',
    description='Visual Servoing ROS2 package',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'parking_controller = visual_servoing.parking_controller:main',
            'cone_detector = visual_servoing.cone_detector:main',
            'cone_sim_marker = visual_servoing.cone_sim_marker:main',
            'homography_transformer = visual_servoing.homography_transformer:main',
            'yolo_annotator = visual_servoing.yolo_annotator:main',
        ],
    },
)
