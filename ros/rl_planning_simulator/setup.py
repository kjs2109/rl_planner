from setuptools import find_packages, setup

package_name = 'rl_planning_simulator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='k',
    maintainer_email='kimjusung2109@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_planning_simulator_node=rl_planning_simulator.rl_planning_simulator_node:main',
            'real_time_simulator=rl_planning_simulator.real_time_simulator:main',
            'rl_mode_trigger=rl_planning_simulator.rl_mode_trigger:main',
        ],
    },
)
