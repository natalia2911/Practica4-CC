
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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
    inputData = assembler.transform(df)
    inputData = inputData.selectExpr('features as features', 'class as label')
    inputData = inputData.select('features', 'label')

    # generate the train/test split.
    (train, test) = inputData.randomSplit([0.8, 0.2])

    # instantiate the base classifier.
    lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)

    # instantiate the One Vs Rest Classifier.
    ovr = OneVsRest(classifier=lr)

    # train the multiclass model.
    ovrModel = ovr.fit(train)

    # score the model on test data.
    predictions = ovrModel.transform(test)

    # obtain evaluator.
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

    # compute the classification error on test data.
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))
    print('Accuracy = ', accuracy)