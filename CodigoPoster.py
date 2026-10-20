# %% =========================
# IMPORTACION DE PAQUETES
# ============================

from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, probplot

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

#Ruta del directorio raiz
BASE_DIR = Path(__file__).resolve().parent

#Ruta de la base de datos
ruta_ex = BASE_DIR / "datos.xlsx"

# %% =========================
# SERIE: EXPORTACIONES DE CARBON DE COLOMBIA (1996-2026)
# ============================

ex_base = pd.read_excel(
    ruta_ex,
    header=None,
    names=["precio_exportaciones"],
    decimal=",",
    sep="\t"
)

# Ver el tipo de objeto de la base de datos (Pandas.DataFrame)
print(type(ex_base))

# Ver primeras y últimas observaciones de la base de datos
print(ex_base.head()) # Primeras observaciones
print(ex_base.tail())
# %%
