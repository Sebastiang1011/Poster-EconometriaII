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
ruta_ex = BASE_DIR / "datos.csv"

# %% =========================
# SERIE: EXPORTACIONES DE CARBON DE COLOMBIA (1996-2026)
# ============================

ex_base = pd.read_csv(
    ruta_ex,
    header=None,
    names=["cantidad_exportaciones"],
    decimal=",",
    sep="\t"
)

# Ver el tipo de objeto de la base de datos (Pandas.DataFrame)
print(type(ex_base))

# Ver primeras y últimas observaciones de la base de datos
print(ex_base.head()) # Primeras observaciones
print(ex_base.tail())
# %%
fechas_ex_base = pd.date_range(
    start="1992-01",
    periods=len(ex_base),
    freq="MS"
)

# Agregar el índice temporal a la base de datos del precio de las exportaciones
ex_base.index = fechas_ex_base

# El tipo de objeto de la base de datos sigue siendo Pandas.DataFrame
print(type(ex_base))

# Ver primeras y últimas observaciones de la base de datos, ahora con índice temporal
print(ex_base.head(21)) # Primeras observaciones
print(ex_base.tail()) # Últimas observaciones 

# %% Creación de la serie de tiempo de "exportaciones"

# La nueva serie de tiempo se va a llamar "expo_serie" y va a tener valores numéricos
expo_serie = ex_base["cantidad_exportaciones"].copy()
expo_serie = pd.to_numeric(expo_serie, errors="coerce")
expo_serie = expo_serie.dropna() # Borrar missing values

# Ver el principio y final de la serie de tiempo
print(expo_serie.head())
print(expo_serie.tail())

# # El tipo de objeto de la base de datos ahora es un Pandas.Series
print(type(expo_serie))

# Ver algunas estadísticas descriptivas de la serie de tiempo 
print(expo_serie.describe())

# %% =========================
# Paso 1: Identificación
# ============================

# Gráfica de la serie de tiempo "exportaciones tradicionales"

plt.figure(figsize=(10, 5))
plt.plot(expo_serie)
plt.title("Exportaciones de carbon - Colombia, 1992-2026")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# %% FAC y FACP de la serie original

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    expo_serie,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC exportaciones de carbon - Colombia")

plot_pacf(
    expo_serie,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de la serie original")

plt.tight_layout()
plt.show()

# %% Tests de Raíz Unitaria 

# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(expo_serie) # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result = kpss(expo_serie, regression="c", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"{nivel}: {valor}")

if kpss_result[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

# Nota: Según los resultados de la prueba ADF y KPSS, NO hay que 
#       diferenciar la serie, dado que esta es estacionaria. 

# %% Logaritmo del precio del carbon (en niveles)

expo_serie_log = np.log(expo_serie)

plt.figure(figsize=(10, 5))
plt.plot(expo_serie_log)
plt.title("Logaritmo de las exportaciones de carbon")
plt.xlabel("Fecha")
plt.ylabel("Log(Exportaciones de carbon)")
plt.grid(True)
plt.show()

# %% FAC y FACP del precio del té (serie en logaritmos)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    expo_serie_log,
    lags=15,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC del logaritmo de las exportaciones de carbon")

plot_pacf(
    expo_serie_log,
    lags=15,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP del logaritmo de las exportaciones de carbon")

plt.tight_layout()
plt.show()

# %% Diferencia del logaritmo

expo_serie_log_diff = np.log(expo_serie).diff().dropna()

# Gráfica de la serie de tiempo de la "diferencia del logaritmo de exportaciones tradicionales"
plt.figure(figsize=(10, 5))
plt.plot(expo_serie_log_diff)
plt.title("Diferencia del logaritmo de la serie original")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# %% FAC y FACP de la diferencia del logaritmo

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    expo_serie_log_diff,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC de la diferencia del logaritmo")

plot_pacf(
    expo_serie_log_diff,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de la diferencia del logaritmo")

plt.tight_layout()
plt.show()

# %% Tests de Raíz Unitaria logaritmo en diferencia de la serie

# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(expo_serie_log_diff) # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result = kpss(expo_serie_log_diff, regression="c", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"{nivel}: {valor}")

if kpss_result[1] < 0.05:
    print("KPSS-logaritmo: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS-logaritmo: no rechazamos H0. Según el test, La serie es estacionaria.")

# Nota: Según los resultados de la prueba ADF y KPSS, NO hay que 
#       diferenciar la serie, dado que esta es estacionaria. 