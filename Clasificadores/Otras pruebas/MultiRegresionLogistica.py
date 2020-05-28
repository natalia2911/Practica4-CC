import sys
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

if __name__ == '__main__':
    conf = SparkConf().setAppName('Practica 4')  

    sc = SparkContext(conf=conf)

    sqlc = SQLContext(sc)

    df = sqlc.read.csv('/user/ccsa15520052/filteredC.small.training', header=True, sep=',',inferSchema=True)

    assembler = VectorAssembler(inputCols=['PSSM_r1_0_A', 'PSSM_r2_-1_S', 'PSSM_central_-1_G','PSSM_r2_-1_Q', 'PSSM_r1_3_E', 'PSSM_r1_-1_E'], outputCol='features')

    training = assembler.transform(df)

    training = training.selectExpr('features as features', 'class as label')

    training.show()

    training = training.select('features', 'label')

    training.show()

    # A partir de aqui es el clasificador
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    #Entrenamos el modelo
    lrModel = lr.fit(training)

    # Pintamos los coeficientes, y los interceptrores
    print("Coefficients: \n" + str(lrModel.coefficientMatrix))
    print("Intercept: " + str(lrModel.interceptVector))

    trainingSummary = lrModel.summary

    # Obtenemos el objetivo por interraccion
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)
        
    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
        % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))