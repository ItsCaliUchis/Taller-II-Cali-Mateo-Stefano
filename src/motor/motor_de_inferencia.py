import joblib
from utils import extractor_reviews
import config
import pandas as pd
from tkinter import Tk, filedialog # abre el exploradorde archivos para cargar el csv
from pathlib import Path
from utils import extractor_reviews # preprocesador
import numpy as np
import torch

def motor(vectorizador, entrada, modelo):

    if isinstance(entrada, str):
        entrada_normalizada = pd.DataFrame(
            {"reviews": [entrada]})

        entrada_individual = True

    elif isinstance(entrada, pd.DataFrame):
        entrada_normalizada = entrada.copy()
        entrada_individual = False

    else:
        raise TypeError("La entrada debe ser un string o un DataFrame")

    entrada_preprocesada = extractor_reviews(entrada_normalizada)
    vectorizar = vectorizador.transform(entrada_preprocesada)
    predicciones = modelo.predict(vectorizar)
    
    if hasattr(modelo, "predict_proba"): # hay algunos modelos que no tienen la clase predict_proba, por eso se chequea

        probabilidades = modelo.predict_proba(vectorizar)

        probabilidades_limpias = [
            {
                str(clase): float(prob)
                for clase, prob in zip(
                    modelo.classes_,
                    np.round(fila, 3)
                )
            }
            for fila in probabilidades
        ]

    else:
        probabilidades_limpias = None

    """ Si la entrada es un string, se ejecuta este bloque
    """
    if entrada_individual:

        predicciones = predicciones[0]
        probabilidades_limpias = (
            probabilidades_limpias[0]
            if probabilidades_limpias is not None
            else None
        )

    else:
        predicciones = predicciones.tolist()

    return {
        "texto": entrada_normalizada,
        "prediccion": predicciones,
        "probabilidades": probabilidades_limpias
    }



def main():
    print("=== INFERENCE ENGINE ===\n")

    print("About: El motor de inferencia predice un sentimiento (negativo, positivo o netrual) a partir del analisis de una review o un dataset con soporte en inglés\n")
    print("En caso de introducir un DataFrame de reviews, la matriz debe ser de orden (n,1) con una columna identificadora de ['reviews']")
    
    print(
        "Modelos disponibles:\n"
        "(1) RNA + BoW (Baseline Unscaled)\n"
        "(2) RNA + BoW (Optimized Unscaled)\n"
        "(3) RNA + TF-IDF Monogram\n"
        "(4) SGD + BoW\n"
        "(5) SGD + TF-IDF Monogram\n"
        "(6) SVM\n"
    )

    try:
        modelo_seleccionado = int(input("Seleccione alguna de las opciones 1-6: "))
    
    except ValueError:

        print("Debe introducir un numero entero")
        return 0

    if modelo_seleccionado not in range(1, 7):
        print("Opción de modelo inválida")
        return 0

    try:
        opcion = int(input("¿Desea analizar una review (1) o cargar un dataset en formato CSV (2)?: "))

    except ValueError:

        print("Debe introducir un numero entero")
        return 0
        

    if opcion == 1:
        
        datos = input("Introduzca su review: ")

    elif opcion == 2:
        
        root = Tk()
        root.withdraw()

        archivo = filedialog.askopenfilename(
            title="Seleccione un dataset",
            filetypes=[("Archivos CSV", "*.csv")]
        )

        root.destroy()

        if archivo:
            datos = pd.read_csv(archivo)
            print("Dataset cargado correctamente")
            print(datos.head())

        else:
            print("No se seleccionó ningún archivo")
            return 0

    else:
        print("Opción de entrada inválida")
        return 0
    
    
    if modelo_seleccionado == 1: # rna + bow baseline unscaled
        
        if (Path(config.bow_vectorizer).exists() 
            and Path(config.rna_bow_baseline_unscaled).exists()):

            vectorizador = joblib.load(config.bow_vectorizer)
            modelo = joblib.load(config.rna_bow_baseline_unscaled)
        
        else:
            print("No se pudo resolver la importacion del vectorizador o el modelo \n"
            "Verifique que sus archivos estan en la ruta correcta")
            return 0

        resultado = motor(vectorizador, datos, modelo)
        print(" === RESULTADOS === ")
        print(resultado["texto"])
        print("\t")
        print(resultado["prediccion"])
        print("\t")
        print(resultado["probabilidades"])
        

    elif modelo_seleccionado == 2: # rna + bow optimized unscaled

        if (Path(config.bow_vectorizer).exists() 
            and Path(config.rna_bow_opt_unscaled).exists()):

            vectorizador = joblib.load(config.bow_vectorizer)
            modelo = joblib.load(config.rna_bow_opt_unscaled)
        
        else:
            print("No se pudo resolver la importacion del vectorizador o el modelo \n"
            "Verifique que sus archivos estan en la ruta correcta")
            return 0

        resultado = motor(vectorizador, datos, modelo)
        print(" === RESULTADOS === ")
        print(resultado["texto"])
        print("\t")
        print(resultado["prediccion"])
        print("\t")
        print(resultado["probabilidades"])


    elif modelo_seleccionado == 3: # rna + tf idf monogram

        if (Path(config.tfidf_monogram_vectorizer).exists() 
            and Path(config.rna_tfidf_monogram).exists()):

            vectorizador = joblib.load(config.tfidf_monogram_vectorizer)
            modelo = joblib.load(config.rna_tfidf_monogram)
        
        else:
            print("No se pudo resolver la importacion del vectorizador o el modelo \n"
            "Verifique que sus archivos estan en la ruta correcta")
            return 0

        resultado = motor(vectorizador, datos, modelo)
        print(" === RESULTADOS === ")
        print(resultado["texto"])
        print("\t")
        print(resultado["prediccion"])
        print("\t")
        print(resultado["probabilidades"])

    elif modelo_seleccionado == 4: # sgd + bow
        
        if (Path(config.bow_vectorizer).exists() 
            and Path(config.sgd_bow).exists()):

            vectorizador = joblib.load(config.bow_vectorizer)
            modelo = joblib.load(config.sgd_bow)
        
        else:
            print("No se pudo resolver la importacion del vectorizador o el modelo \n"
            "Verifique que sus archivos estan en la ruta correcta")
            return 0

        resultado = motor(vectorizador, datos, modelo)
        print(" === RESULTADOS === ")
        print(resultado["texto"])
        print("\t")
        print(resultado["prediccion"])
        print("\t")
        print(resultado["probabilidades"])

    elif modelo_seleccionado == 5: # sgd + tf idf monogram

        if (Path(config.tfidf_monogram_vectorizer).exists() 
            and Path(config.sgd_tfidf_monogram).exists()):

            vectorizador = joblib.load(config.tfidf_monogram_vectorizer)
            modelo = joblib.load(config.sgd_tfidf_monogram)
        
        else:
            print("No se pudo resolver la importacion del vectorizador o el modelo \n"
            "Verifique que sus archivos estan en la ruta correcta")
            return 0

        resultado = motor(vectorizador, datos, modelo)
        print(" === RESULTADOS === ")
        print(resultado["texto"])
        print("\t")
        print(resultado["prediccion"])
        print("\t")
        print(resultado["probabilidades"])

    elif modelo_seleccionado == 6: # svm + bow

        if (Path(config.bow_vectorizer).exists() 
            and Path(config.svm_bow).exists()):

            vectorizador = joblib.load(config.bow_vectorizer)
            modelo = joblib.load(config.svm_bow)
        
        else:
            print("No se pudo resolver la importacion del vectorizador o el modelo \n"
            "Verifique que sus archivos estan en la ruta correcta")
            return 0

        resultado = motor(vectorizador, datos, modelo)
        print(" === RESULTADOS === ")
        print(resultado["texto"])
        print("\t")
        print(resultado["prediccion"])
        print("\t")
        print(resultado["probabilidades"])

if __name__ == "__main__":
    main()