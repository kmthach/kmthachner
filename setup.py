from setuptools import find_namespace_packages, find_packages, setup

VERSION = "0.0.1"
DESCRIPTION = "An NER API trained on CONLL2003"
LONG_DESCRIPTION = "An NER API trained on CONLL2003"

# Setting up
setup(
    name="kmthachner",
    version=VERSION,
    author="Nguyen Kim Thach",
    author_email="<kmthach.ai@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_namespace_packages(),
    include_package_data=True,
    package_data={"": ["*.txt",'*.json','*.w']},
    install_requires=["torch", "seqeval", "datasets"],
)
