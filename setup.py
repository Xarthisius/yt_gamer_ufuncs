import os


def configuration(parent_package="", top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration("yt_gamer_ufuncs", parent_package, top_path)
    config.add_extension("fields", ["yt_gamer_ufuncs/fields.c"])
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(configuration=configuration)
    package_name = "yt_gamer_ufuncs"
    version = "0.1"
    README = os.path.join(os.path.dirname(__file__), "README.md")
    long_description = open(README).read()
    setup(
        name=package_name,
        version=version,
        configuration=configuration,
        description=("Gamer fields rewritten as ufuncs"),
        long_description=long_description,
        classifiers=[
            "Programming Language :: Python",
            ("Topic :: Software Development :: Libraries :: Python Modules"),
        ],
        keywords="data",
        author="Kacper Kowalik <xarthisius.kk@gmail.com>",
        license="BSD",
        package_dir={package_name: package_name},
        packages=[package_name],
        install_requires=["numpy"],
        url="https://yt-project.org/",
        author_email="xarthisius.kk@gmail.com",
    )
