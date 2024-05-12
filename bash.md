# Apache vs The Fraudster

After handling configurations of Apache Hadoop, Hive and Spark, all of these applications started and "creditcard.csv" file uploaded to HDFS.

<pre><code>$ jps</code></pre>
```
4033 DataNode
7490 ExecutorLauncher
7332 SparkSubmit
4533 ResourceManager
7605 Jps
4296 SecondaryNameNode
3865 NameNode
7546 YarnCoarseGrainedExecutorBackend
4700 NodeManager
5919 RunJar
```


<pre><code>$ hdfs dfs -put /home/arda/Downloads/creditcard.csv /</code></pre>

Created a database and a table following the uploading on Hive.

<pre><code>hive > CREATE DATABASE apache_vs_fraudster;</code></pre>

<pre><code>hive > CREATE EXTERNAL TABLE fraud_detection (
    `Time` double,
    V1 double,
    V2 double,
    V3 double,
    V4 double,
    V5 double,
    V6 double,
    V7 double,
    V8 double,
    V9 double,
    V10 double,
    V11 double,
    V12 double,
    V13 double,
    V14 double,
    V15 double,
    V16 double,
    V17 double,
    V18 double,
    V19 double,
    V20 double,
    V21 double,
    V22 double,
    V23 double,
    V24 double,
    V25 double,
    V26 double,
    V27 double,
    V28 double,
    AMOUNT double,
    CLASS int
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ',';
</code></pre>

Load "creditcard.csv" into "fraud_detection" table:

<pre><code>hive > LOAD DATA INPATH '/creditcard.csv' INTO TABLE fraud_detection;</code></pre>

Run PySpark with .py script

<pre><code> $ ./spark-submit --master yarn --queue dev /home/hadoop/fraud_script.py</code></pre>
