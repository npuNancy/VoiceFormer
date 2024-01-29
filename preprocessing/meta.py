import pandas as pd
import numpy as np
import pathlib
import random
import sys
import os
from enum import Enum, unique


@unique
class SensorType(Enum):
    acc = 0
    gyr = 1


if __name__ == "__main__":
    print(SensorType.acc.name)
