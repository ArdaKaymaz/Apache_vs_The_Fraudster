![apache_fraudster](https://github.com/ArdaKaymaz/Apache_vs_The_Fraudster/assets/146623362/f543c25f-5b6f-4273-ab58-2509b3c67e81)

# Apache vs The Fraudster

<strong>Abstract</strong>

This project covers a comprehensive case study on fraud detection utilizing big data technologies, specifically Apache Hadoop, Hive, and Spark. The project aims to develop a robust system for detecting fraudulent activities in large datasets. To achieve this, specific versions of Apache Hadoop (3.2.1), Hive (3.1.2), and Spark (3.2.1) were employed to ensure stability and efficiency. The process involved distributing fraud detection data using Apache Hadoop, creating structured tables with Apache Hive, and implementing machine learning algorithms for fraud detection using Apache Spark. In addition to this, significant differences was found in ML algorithms' results between Spark 3.2.1 and Spark 3.5.1.

<strong>Introduction</strong>

Fraud detection is a critical aspect of data analytics, particularly in industries such as finance, insurance, and e-commerce. With the exponential growth of data, traditional fraud detection methods have become inadequate in handling large-scale datasets. Big data technologies offer scalable solutions to address this challenge by leveraging distributed computing frameworks. In this study, we explore the integration of Apache Hadoop, Hive, and Spark to develop a robust fraud detection system.

<strong>Methodology</strong>

Data Preparation: The project begins with preparing the fraud detection dataset for analysis. Raw data is distributed across the Hadoop cluster to enable parallel processing and efficient data storage.

Data Processing with Hive: Apache Hive is utilized to create structured tables and perform data transformation tasks. Hive's SQL-like query language (HiveQL) simplifies the process of querying and analyzing large datasets stored in Hadoop Distributed File System (HDFS).

Machine Learning Model Development with Spark: Apache Spark is employed for building machine learning models to detect fraudulent activities. Spark's MLlib library provides a wide range of algorithms and tools for scalable machine learning tasks. In this study, various supervised and unsupervised learning algorithms are explored to identify fraudulent patterns within the dataset.

Model Evaluation and Deployment: The developed machine learning models are evaluated using appropriate performance metrics such as precision, recall, and F1-score. Once the models demonstrate satisfactory performance, they are deployed into production environments for real-time or batch processing of incoming data streams.
