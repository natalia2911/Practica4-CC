from pyspark import SparkContext, SparkConf, SQLContext
import sys

if __name__ == '__main__':

    conf = SparkConf().setAppName("Natalia Martir - Practica 4")
    Sp = SparkContext(conf=conf)

    headers = Sp.textFile("/user/datasets/ecbdl14/ECBDL14_IR2.header").collect()
    headers = list(filter(lambda x: "@inputs" in x, headers))[0]
    headers = headers.replace(",", "").strip().split()
    del headers[0]
    headers.append("class")

    sqlc = SQLContext(Sp)
    df = sqlc.read.csv('/user/datasets/ecbdl14/ECBDL14_IR2.data', header=False, inferSchema=True)

    for i, colname in enumerate(df.columns):
        df = df.withColumnRenamed(colname, headers[i])

    df = df.select("PSSM_r1_0_A", "PSSM_r2_-1_S", "PSSM_central_-1_G","PSSM_r2_-1_Q", "PSSM_r1_3_E", "PSSM_r1_-1_E", "class")
    df.write.csv('./filteredC.small.training', header=True)



