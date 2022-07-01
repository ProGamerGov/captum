#!/usr/bin/env python3
import captum.attr as attr  # noqa
import captum.concept as concept  # noqa
import captum.influence as influence  # noqa

try:
    import captum.insights as insights  # noqa
except (ImportError, AssertionError):
    print(
        "The {} libraries are required if using Captum's Insights module".format(
            ["flask", "ipython", "ipywidgets", "jupyter", "flask-compress"]
        )
    )
import captum.log as log  # noqa
import captum.metrics as metrics  # noqa
import captum.robust as robust  # noqa


__version__ = "0.5.0"
