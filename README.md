# DeteccionDeDisparos
Roles:
- Leonardo -> Lider en Machine Learning
- Miguel -> Desarrollador de Interfaz / Prototipo
- Jhoan -> Responsable de Pruebas e Integración
- Jesus -> Responsable de datos (DataSet)
- Xiomara -> Responsable de Documentación Técnica

# 📌 Fases del Proyecto y Roles

## 🧠 Fase 1: Planificación y Definición del Proyecto
**Descripción:**
Se define el problema, los objetivos del sistema y el alcance del proyecto. Además, se establece la estructura del repositorio y la organización del equipo.

**Actividades:**
- Definir el problema y objetivo del sistema
- Establecer objetivos específicos
- Seleccionar tecnologías
- Crear estructura del repositorio
- Asignar roles del equipo

**Roles participantes:**
- Líder de Machine Learning  
- Responsable de Datos (Dataset)  
- Responsable de Documentación Técnica  
- Desarrollador de Interfaz / Prototipo  
- Responsable de Pruebas e Integración  

**Resultado:**
- Proyecto definido y organizado
- Repositorio inicial creado

---

## 📊 Fase 2: Recolección y Organización del Dataset
**Descripción:**
Se recopilan audios de disparos y no disparos, y se organizan adecuadamente.

**Actividades:**
- Buscar y descargar audios
- Clasificar audios por categorías
- Verificar formato y calidad
- Documentar fuentes del dataset

**Roles participantes:**
- Responsable de Datos (Dataset)  
- Líder de Machine Learning  
- Responsable de Documentación Técnica  

**Resultado:**
- Dataset organizado y documentado

---

## 🔍 Fase 3: Exploración y Análisis de Datos
**Descripción:**
Se analizan los audios para entender sus características y detectar posibles problemas.

**Actividades:**
- Cargar audios
- Analizar duración y frecuencia
- Visualizar formas de onda
- Detectar errores o inconsistencias

**Roles participantes:**
- Responsable de Datos (Dataset)  
- Líder de Machine Learning  
- Responsable de Documentación Técnica  
- Responsable de Pruebas e Integración  

**Resultado:**
- Conocimiento del dataset
- Identificación de problemas iniciales

---

## 🎧 Fase 4: Preprocesamiento de Audio
**Descripción:**
Se preparan los audios para que puedan ser utilizados por el modelo.

**Actividades:**
- Normalizar audios
- Ajustar sample rate
- Recortar o estandarizar duración
- Limpiar datos

**Roles participantes:**
- Líder de Machine Learning  
- Responsable de Pruebas e Integración  

**Resultado:**
- Audios listos para procesamiento

---

## 🧬 Fase 5: Extracción de Características
**Descripción:**
Se convierten los audios en datos numéricos mediante características acústicas.

**Actividades:**
- Extraer MFCC
- Calcular Zero Crossing Rate
- Calcular energía (RMS)
- Obtener spectral centroid
- Construir dataset tabular

**Roles participantes:**
- Líder de Machine Learning  
- Responsable de Pruebas e Integración  
- Responsable de Documentación Técnica  

**Resultado:**
- Dataset listo para entrenamiento

---

## 🤖 Fase 6: Entrenamiento del Modelo
**Descripción:**
Se entrenan modelos de machine learning para clasificar los audios.

**Actividades:**
- Dividir datos en entrenamiento y prueba
- Entrenar modelos (Logistic Regression, Random Forest)
- Comparar resultados
- Guardar modelo final

**Roles participantes:**
- Líder de Machine Learning  
- Responsable de Pruebas e Integración  
- Responsable de Documentación Técnica  

**Resultado:**
- Modelo entrenado y seleccionado

---

## 📈 Fase 7: Evaluación del Modelo
**Descripción:**
Se mide el rendimiento del modelo y se analizan sus resultados.

**Actividades:**
- Calcular métricas (accuracy, precision, recall, F1)
- Generar matriz de confusión
- Comparar modelos
- Analizar resultados

**Roles participantes:**
- Líder de Machine Learning  
- Responsable de Documentación Técnica  
- Responsable de Pruebas e Integración  

**Resultado:**
- Evaluación completa del sistema

---

## 🌐 Fase 8: Desarrollo del Prototipo
**Descripción:**
Se implementa una interfaz para interactuar con el sistema.

**Actividades:**
- Crear interfaz (Streamlit)
- Permitir carga de audio
- Conectar modelo con la interfaz
- Mostrar predicción

**Roles participantes:**
- Desarrollador de Interfaz / Prototipo  
- Líder de Machine Learning  
- Responsable de Pruebas e Integración  

**Resultado:**
- Prototipo funcional

---

## 🛠️ Fase 9: Pruebas e Integración
**Descripción:**
Se valida el correcto funcionamiento del sistema completo.

**Actividades:**
- Probar todos los módulos
- Verificar ejecución en diferentes entornos
- Corregir errores
- Integrar componentes

**Roles participantes:**
- Responsable de Pruebas e Integración  
- Líder de Machine Learning  
- Desarrollador de Interfaz / Prototipo  
- Responsable de Datos (Dataset)  

**Resultado:**
- Sistema estable e integrado

---

## 📚 Fase 10: Documentación Final y Cierre
**Descripción:**
Se completa la documentación del proyecto y se prepara para la entrega.

**Actividades:**
- Completar README
- Documentar metodología y arquitectura
- Redactar conclusiones
- Revisar el repositorio

**Roles participantes:**
- Responsable de Documentación Técnica  
- Líder de Machine Learning  
- Responsable de Datos (Dataset)  
- Desarrollador de Interfaz / Prototipo  
- Responsable de Pruebas e Integración  

**Resultado:**
- Proyecto final documentado y listo para presentación