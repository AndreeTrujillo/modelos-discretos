import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Título de la aplicación
st.title('Modelado del Rendimiento de Cultivos usando Regresión Lineal Múltiple')

# Cargar datos
st.sidebar.header('Cargar archivo de datos CSV')
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Mostrar datos cargados
    st.write("Datos cargados:")
    st.write(df.head())
    
    # Filtrar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Seleccionar variables predictoras y la variable objetivo
    st.sidebar.header('Selección de variables')
    predictores = st.sidebar.multiselect('Selecciona las variables predictoras', numeric_cols)
    objetivo = st.sidebar.selectbox('Selecciona la variable objetivo', numeric_cols)

    if predictores and objetivo:
        X = df[predictores]
        y = df[objetivo]

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluar el rendimiento del modelo
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

        # Mostrar métricas de evaluación con interpretaciones
        st.subheader('Evaluación del Modelo')
        st.write(f"**Error Cuadrático Medio (MSE)**: {mse:.2f}")
        st.write("El MSE mide el promedio de los errores al cuadrado, indicando qué tan cerca están las predicciones de los valores reales. Un MSE más bajo indica un mejor rendimiento del modelo.")

        st.write(f"**Coeficiente de Determinación (R²)**: {r2:.2f}")
        st.write("El R² indica la proporción de la varianza en la variable objetivo que es explicada por las variables predictoras. Un R² de 1.00 significa que el modelo explica toda la variabilidad de los datos, lo cual es ideal pero raro en la práctica.")

        st.write(f"**Coeficiente de Determinación Ajustado (R² ajustado)**: {adjusted_r2:.2f}")
        st.write("El R² ajustado es una versión del R² que ha sido ajustada por el número de variables predictoras. Es útil para comparar modelos con diferentes números de variables predictoras.")

        # Mostrar coeficientes del modelo
        coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coeficiente'])
        st.subheader('Coeficientes del Modelo')
        st.write(coef_df)
        st.write(f"**Intercepto**: {model.intercept_}")
        st.write("El intercepto es el valor esperado de la variable objetivo cuando todas las variables predictoras son cero.")

        # Gráfico de dispersión de predicciones vs valores reales
        st.subheader('Gráfico de Predicciones vs. Valores Reales')
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel('Valores Reales')
        plt.ylabel('Predicciones')
        plt.title('Predicciones vs. Valores Reales')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)  # Línea de referencia
        st.pyplot(plt.gcf())

        # Interpretación del gráfico
        st.write("**Interpretación del Gráfico**")
        st.write("""
        El gráfico de dispersión muestra la relación entre los valores reales y las predicciones del modelo.
        - **Ejes**: El eje x representa los valores reales y el eje y las predicciones.
        - **Puntos**: Cada punto representa una observación individual.
        - **Línea de referencia**: La línea discontinua (negra) indica la línea ideal donde las predicciones serían exactamente iguales a los valores reales (y=x).
        
        Si los puntos están cerca de la línea discontinua, significa que el modelo predice bien los valores reales. Una gran dispersión alrededor de la línea indicaría un modelo con menor precisión.
        """)

        # Predicción para nuevos datos
        st.sidebar.header('Predicción para nuevos datos')
        nuevos_datos = {var: st.sidebar.number_input(f'Ingresar {var}', value=0.0) for var in predictores}

        if st.sidebar.button('Predecir'):
            nuevos_datos_df = pd.DataFrame([nuevos_datos])
            prediccion = model.predict(nuevos_datos_df)
            st.subheader('Predicción del Rendimiento')
            st.write(f"**Predicción del rendimiento**: {prediccion[0]:.2f}")
else:
    st.write("Por favor, sube un archivo CSV para comenzar.")
