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

  # A partir de aquí es el clasifi
  training = assembler.transform(df)

  training = training.selectExpr('features as features', 'class as label')

  training.show()

  training = training.select('features', 'label')

  training.show()

  lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

  # Fit the model
  lrModel = lr.fit(training)

  # Print the coefficients and intercept for logistic regression
  print('Coefficients: ' + str(lrModel.coefficients))
  print('Intercept: ' + str(lrModel.intercept))


  sc.stop()
