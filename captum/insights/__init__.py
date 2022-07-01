from warnings import warn

try:
    from captum.insights.attr_vis import AttributionVisualizer, Batch  # noqa
except (ImportError, AssertionError) as e:
    warning_msg = (
        "Missing required libraries for Captum's Insights module."
        + " This warning can be ignored if not using Insights."
        + " {}".format(e)
    )
    warn(warning_msg)
