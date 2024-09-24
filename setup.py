from setuptools import find_packages, setup

package_name = 'segmentation_pkg'

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
    maintainer='chenchangtai',
    maintainer_email='chenchangtai@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'segmentation_node = segmentation_pkg.segmentation_node_v2:main',
            'segmentation_node_RViz = segmentation_pkg.segmentation_node_RViz:main',
        ],
    },
)
