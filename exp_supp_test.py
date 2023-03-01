import os
import pandas as pd
from src.start import classifier

result = classifier.searching("для учетной политике по каким ФСБУ были изменения", 6, 0.8)
print(result)