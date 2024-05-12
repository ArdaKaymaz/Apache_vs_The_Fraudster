# Importing The Libraries and Building The Spark Session

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Apache vs The Fraudster").getOrCreate()

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import VectorAssembler, MaxAbsScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Reading The Data

df = spark.read.format("csv").option("header", True).option("separator", ",").option("inferSchema", True).load("hdfs:///user/hive/warehouse/apache_vs_fraudster.db/fraud_detection")

# Preparing The Data for ML

X = [col for col in df.columns if col not in ["Time", "Class"]]
y = df.select("Class")

vectorizer = VectorAssembler(inputCols=X, outputCol="independent_features")

df_vectorized = vectorizer.transform(df)

df_final = df_vectorized.select("independent_features", "Class")

df_final_class_1 = df_final.filter(df_final["Class"] == 1)
df_final_class_0 = df_final.filter(df_final["Class"] == 0)


# Train-Test Split

df_sample_class_1 = df_final_class_1.sample(False, 0.8, seed=123)
df_sample_class_0 = df_final_class_0.sample(False, 0.8, seed=123)
df_train = df_sample_class_1.union(df_sample_class_0)
df_test = df_final.subtract(df_train)

scaler = MaxAbsScaler(inputCol="independent_features",
                       outputCol="scaled_ind_features")

scaler_model = scaler.fit(df_final)
df_train = scaler_model.transform(df_train).drop("independent_features").select(["scaled_ind_features", "Class"])
df_test = scaler_model.transform(df_test).drop("independent_features").select(["scaled_ind_features", "Class"])


# Model Functions

def logistic_regression(train_data, test_data):
    train_set = train_data.alias("train_data")
    test_set = test_data.alias("test_data")

    lr = LogisticRegression(featuresCol='scaled_ind_features', labelCol='Class')
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, 
            [
            lr.getRegParam(),
            lr.getRegParam() + 0.01,
            lr.getRegParam() + 0.05,
            lr.getRegParam() + 0.1
            ]) \
        .addGrid(lr.elasticNetParam, 
            [
            lr.getElasticNetParam(),
            lr.getElasticNetParam() + 0.25,
            lr.getElasticNetParam() + 0.5,
            lr.getElasticNetParam() + 0.75
            ]) \
        .addGrid(lr.maxIter, 
            [
            lr.getMaxIter(),
            int(lr.getMaxIter()*0.80),
            int(lr.getMaxIter()*0.90),
            int(lr.getMaxIter()*1.10),
            int(lr.getMaxIter()*1.20)
            ]) \
        .addGrid(lr.tol, 
            [
            lr.getTol(),
            lr.getTol()*0.80,
            lr.getTol()*0.90,
            lr.getTol()*1.10,
            lr.getTol()*1.20
            ]) \
        .addGrid(lr.threshold, 
            [
            lr.getThreshold(),
            lr.getThreshold()*0.90,
            lr.getThreshold()*1.10
            ]) \
        .build()

    evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR", rawPredictionCol="rawPrediction", labelCol="Class")
        
    cv = CrossValidator(estimator=lr,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator)
        
    model = cv.fit(train_set).bestModel

    predictions = model.transform(test_set)

    test_metrics = test_metric_calculator(predictions)

    train_metrics = train_metric_calculator(model)

    return model, train_metrics, test_metrics, predictions
    
    
def random_forest(train_data, test_data):
    train_set = train_data.alias("train_data")
    test_set = test_data.alias("test_data")

    rf = RandomForestClassifier(featuresCol='scaled_ind_features', labelCol='Class')
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, 
            [
            rf.getMaxDepth(),
            int(rf.getMaxDepth()*1.5),
            int(rf.getMaxDepth()*2),
            int(rf.getMaxDepth()*3)
            ]) \
        .addGrid(rf.numTrees,
            [
            rf.getNumTrees(),
            int(rf.getNumTrees()*0.8),
            int(rf.getNumTrees()*1.25),
            int(rf.getNumTrees()*1.3),
            int(rf.getNumTrees()*1.5)
            ]) \
        .build()

    evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR", rawPredictionCol="rawPrediction", labelCol="Class")
        
    cv = CrossValidator(estimator=rf,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator)
        
    model = cv.fit(train_set).bestModel
    predictions = model.transform(test_set)

    test_metrics = test_metric_calculator(predictions)

    train_metrics = train_metric_calculator(model)

    return model, train_metrics, test_metrics, predictions

            
    
def train_metric_calculator(model): 
    summary = model.summary

    train_metrics = {}

    for i, rate in enumerate(summary.falsePositiveRateByLabel):
        train_metrics[f"False Positive Rate (Label {i})"] = rate

    for i, rate in enumerate(summary.truePositiveRateByLabel):
        train_metrics[f"True Positive Rate (Label {i})"] = rate

    for i, prec in enumerate(summary.precisionByLabel):
        train_metrics[f"Precision (Label {i})"] = prec

    for i, rec in enumerate(summary.recallByLabel):
        train_metrics[f"Recall (Label {i})"] = rec

    for i, f in enumerate(summary.fMeasureByLabel()):
        train_metrics[f"F-measure (Label {i})"] = f

    train_metrics['Accuracy'] = summary.accuracy
    train_metrics['Weighted False Positive Rate'] = summary.weightedFalsePositiveRate
    train_metrics['Weighted True Positive Rate'] = summary.weightedTruePositiveRate
    train_metrics['Weighted F-measure'] = summary.weightedFMeasure()
    train_metrics['Weighted Precision'] = summary.weightedPrecision
    train_metrics['Weighted Recall'] = summary.weightedRecall

    return train_metrics
    

def test_metric_calculator(predictions):
    test_metrics = {}

    true_positive = predictions.filter("prediction = 1 AND Class = 1").count()
    false_positive = predictions.filter("prediction = 1 AND Class = 0").count()
    false_negative = predictions.filter("prediction = 0 AND Class = 1").count()
    true_negative = predictions.filter("prediction = 0 AND Class = 0").count()

    test_metrics['true_positive'] = true_positive
    test_metrics['false_positive'] = false_positive
    test_metrics['false_negative'] = false_negative
    test_metrics['true_negative'] = true_negative

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0

    test_metrics['precision'] = precision
    test_metrics['recall'] = recall

    return test_metrics


# Model Training

## Logistic Regression

lr_model, lr_train_metrics, lr_test_metrics, lr_predictions = logistic_regression(df_train, df_test)


## Random Forest

rf_model, rf_train_metrics, rf_test_metrics, rf_predictions = random_forest(df_train, df_test)


# Comparing The Results

print("########## Comparing The Results ##########")

## Results of Logistic Regression

print("##### Train Metrics of Logistic Regression #####")
for key, value in lr_train_metrics.items():
    print(f"{key}: {value}")

print("\n### Test Metrics of Logistic Regression ###")
for key, value in lr_test_metrics.items():
    print(f"{key}: {value}")

## Best Hyperparameters for Logistic Regression

lr_params = ['regParam', 'elasticNetParam', 'maxIter', 'tol', 'threshold']

print("### Best Hyperparameters for Logistic Regression ###")

for param_name in lr_params:
    param_value = lr_model.getOrDefault(param_name)
    print(f"{param_name}: {param_value}")


## Results of Random Forest

print("##### Train Metrics of Random Forest #####")
for key, value in rf_train_metrics.items():
    print(f"{key}: {value}")

print("\n### Test Metrics of Random Forest ###")
for key, value in rf_test_metrics.items():
    print(f"{key}: {value}")

## Best Hyperparameters for Random Forest

rf_params = ["maxDepth", "numTrees"]

print("### Best Hyperparameters for Random Forest ###")

for param_name in rf_params:
    param_value = rf_model.getOrDefault(param_name)
    print(f"{param_name}: {param_value}")
