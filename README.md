![apache_fraudster](https://github.com/ArdaKaymaz/Apache_vs_The_Fraudster/assets/146623362/f543c25f-5b6f-4273-ab58-2509b3c67e81)

# Apache vs The Fraudster

## Abstract

This project covers a comprehensive case study on fraud detection utilizing big data technologies, specifically Apache Hadoop, Hive, and Spark. The project aims to develop a robust system for detecting fraudulent activities in large datasets. To achieve this, specific versions of Apache Hadoop (3.2.1), Hive (3.1.2), and Spark (3.2.1) were employed to ensure stability and efficiency. The process involved distributing fraud detection data using Apache Hadoop, creating structured tables with Apache Hive, and implementing machine learning algorithms for fraud detection using Apache Spark. In addition to this, significant differences was found in ML algorithms' results between Spark 3.2.1 and Spark 3.5.1.


## Introduction

Fraud detection is a critical aspect of data analytics, particularly in industries such as finance, insurance, and e-commerce. With the exponential growth of data, traditional fraud detection methods have become inadequate in handling large-scale datasets. Big data technologies offer scalable solutions to address this challenge by leveraging distributed computing frameworks. In this study, we explore the integration of Apache Hadoop, Hive, and Spark to develop a robust fraud detection system.


## Methodology

### Data Preparation
The project begins with preparing the fraud detection dataset for analysis. Raw data is distributed across the Hadoop cluster to enable parallel processing and efficient data storage.

![hdfs](https://github.com/ArdaKaymaz/Apache_vs_The_Fraudster/assets/146623362/4fbafb33-b558-4288-9f40-250f3c785d93)

### Data Processing with Hive
Apache Hive is utilized to create structured tables and perform data transformation tasks. Hive's SQL-like query language (HiveQL) simplifies the process of querying and analyzing large datasets stored in Hadoop Distributed File System (HDFS).

### Machine Learning Model Development with Spark
Apache Spark is employed for building machine learning models to detect fraudulent activities. Spark's MLlib library provides a wide range of algorithms and tools for scalable machine learning tasks. In this study, various supervised and unsupervised learning algorithms are explored to identify fraudulent patterns within the dataset.

### Model Evaluation
The developed machine learning models are evaluated using appropriate performance metrics such as precision, recall, and F1-score.


## Results and Discussion

The integration of Apache Hadoop, Hive, and Spark offers a scalable and efficient solution for fraud detection in large datasets. The utilization of distributed computing frameworks enables parallel processing of data, reducing computation time and improving overall system performance. The machine learning models developed using Spark exhibit promising results in detecting fraudulent activities, achieving high accuracy and reliability.

### Logistic Regression Results in Spark 3.2.1
![logistic_regression_pyspark](https://github.com/ArdaKaymaz/Apache_vs_The_Fraudster/assets/146623362/4fadc73f-8583-4e29-8ca8-7fd9ba3c56c6)

### Logistic Regression Results in Spark 3.5.1
![logistic_regression](https://github.com/ArdaKaymaz/Apache_vs_The_Fraudster/assets/146623362/7a4c1d2e-a4cb-4928-b8db-1fd7ed07c0cd)

### Random Forest Results in Spark 3.2.1
![random_forest_pyspark](https://github.com/ArdaKaymaz/Apache_vs_The_Fraudster/assets/146623362/ea8d25e4-4e49-48f7-a287-3b8b421a0e9a)

### Random Forest Results in Spark 3.5.1
![random_forest](https://github.com/ArdaKaymaz/Apache_vs_The_Fraudster/assets/146623362/842c0235-68e1-44a2-8898-b79b08c34cd2)

While the test recall value of logistic regression was resulting as <strong>~51%</strong> and <strong>60%</strong>, in Spark 3.2.1 and Spark 3.5.1, respectively; the test recall value of random forest was found as <strong>~65%</strong> and <strong>78%</strong>, in Spark 3.2.1 and Spark 3.5.1, respectively. It is evident that there is a significant difference between the results of Spark 3.2.1 and Spark 3.5.1 versions in both algorithms and random forest resulted higher in both precision and recall. In the context of banking, it is more important to detect fraudulent transactions and in this case, the recall value of random forest algorithm sounds promising with 78% recall value.

## Conclusion
In conclusion, this case study demonstrates the effectiveness of big data technologies, specifically Apache Hadoop, Hive, and Spark, in developing a robust fraud detection system. By leveraging distributed computing and machine learning algorithms, organizations can enhance their ability to identify and prevent fraudulent activities in large-scale datasets. Versions of the big data technologies affects the performance of algorithms used in machine learning process and the versions of big data technologies should be considered comprehensively.
