from setuptools import setup, find_packages

setup(
    name="llm-router-service",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tiktoken",
        "textstat",
        "promptlayer",
        "litellm",
        "python-dotenv",
        "langchain-ollama",
        "requests",
        "pydantic",
        "tqdm",
    ],
    python_requires=">=3.10",
    author="Srihari Raman",
    author_email="sriharii@fyrassolutions.com",
    description="LLM router with councils and selectors for intelligent model selection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/Fyras-Solutions-Org/LLMRouterService",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
