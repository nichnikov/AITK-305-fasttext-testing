import os
import pandas as pd
from src.start import classifier

result = classifier.searching("кто может сдавать упрощенный баланс", 6, 0.8)
print(result)