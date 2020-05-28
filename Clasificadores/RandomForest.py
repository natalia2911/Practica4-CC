from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import sys

if __name__ == '__main__':
    conf = SparkConf().setAppName("Practica 4")
    sc = SparkContext(conf=conf)

    #Leemos el fichero de cvs, alojado en nuestro hdfs
    sqlc = SQLContext(sc)
    df = sqlc.read.csv('/user/ccsa15520052/filteredC.small.training', header=True, sep=',',inferSchema=True)

    #Leemos nuestras columnas
    assembler = VectorAssembler(inputCols=['PSSM_r1_0_A', 'PSSM_r2_-1_S', 'PSSM_central_-1_G','PSSM_r2_-1_Q', 'PSSM_r1_3_E', 'PSSM_r1_-1_E'], outputCol='features')
    
    #Las trasnformamos en features y label
    data = assembler.transform(df)
    data = data.selectExpr('features as features', 'class as label')
    data = data.select('features', 'label')

    #A partir de aqui es el clasificador
    #Ajustamos el conjunto de datos para incluir las etiquetas en el indice
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    #Carasteristicas categoricas, las cuales tenemos que indexar.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    #Dividimos el conjunto de datos, en prueba y entrenamiento.
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    #Entrenamos el modelo
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

    #Convertimos indexadores en etiquetas originales
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    #Entrenamos el modelo
    model = pipeline.fit(trainingData)

    #Hacemos las predicciones
    predictions = model.transform(testData)

    predictions.select("predictedLabel", "label", "features").show(5)

    #Evaluamos el modelo
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))
    print('Accuracy = ', accuracy)

    rfModel = model.stages[2]
    print(rfModel)  # summary only