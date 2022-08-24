import pathlib
import setuptools

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setuptools.setup(
    name="pytorch-bonfire",
    version="0.1.0",
    py_modules=['pytorch-bonfire'],
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/JAEarly/Bonfire",
    author="Joseph Early",
    author_email="joseph.early.ai@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(include=['bonfire']),
)
