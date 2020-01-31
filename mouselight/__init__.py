import warnings

import mouselight.utils

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("always", category=UserWarning)


__version__ = "0.0.1"
