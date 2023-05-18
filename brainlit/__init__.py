import warnings

import brainlit.algorithms
import brainlit.BrainLine
import brainlit.feature_extraction
import brainlit.preprocessing
import brainlit.utils
import brainlit.viz

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("always", category=UserWarning)


__version__ = "0.3.4"
