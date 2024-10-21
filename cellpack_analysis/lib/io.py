import numpy as np


def format_time(seconds):
    """Format time in seconds to a human readable format."""
    if seconds == np.inf:
        return "∞"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"
