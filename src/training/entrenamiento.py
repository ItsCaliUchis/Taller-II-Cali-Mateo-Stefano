import pandas as pd
import config
from utils import ejecutar_etapa, verificar_embeddings
from pathlib import Path
import subprocess
import sys
import numpy as np

def main():

    X_bow, X_bigram, X_monogram = verificar_embeddings()

if __name__ == "__main__":
    main()