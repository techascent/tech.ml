# tech.ml

[![Clojars Project](https://img.shields.io/clojars/v/techascent/tech.ml.svg)](https://clojars.org/techascent/tech.ml)

Library to encapsulate a few core concepts of techascent system.

* [API Documentation](https://techascent.github.io/tech.ml/)



## Simple Regression And Classification

We start out a require:

```clojure
user> (require '[tech.v3.dataset :as ds])
```

And move to a dataset.  We will use the famous Iris dataset:


```clojure
user> (def ds (ds/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv"))
#'user/ds
user> (ds/head ds)
https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv [5 5]:

| sepal_length | sepal_width | petal_length | petal_width | species |
|--------------|-------------|--------------|-------------|---------|
|          5.1 |         3.5 |          1.4 |         0.2 |  setosa |
|          4.9 |         3.0 |          1.4 |         0.2 |  setosa |
|          4.7 |         3.2 |          1.3 |         0.2 |  setosa |
|          4.6 |         3.1 |          1.5 |         0.2 |  setosa |
|          5.0 |         3.6 |          1.4 |         0.2 |  setosa |
```

### Preparing The Dataset

We need to have all numeric columns in our dataset.  The species column is a categorical
column and we will need to convert it to a numeric column while remembering
what mapping we used.  We introduce the column filters namespace that performs
various filtering operations on the columns themselves returning new datasets:

```clojure
user> (require '[tech.v3.dataset.column-filters :as cf])
nil
user> (ds/head (cf/numeric ds))
https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv [5 4]:

| sepal_length | sepal_width | petal_length | petal_width |
|--------------|-------------|--------------|-------------|
|          5.1 |         3.5 |          1.4 |         0.2 |
|          4.9 |         3.0 |          1.4 |         0.2 |
|          4.7 |         3.2 |          1.3 |         0.2 |
|          4.6 |         3.1 |          1.5 |         0.2 |
|          5.0 |         3.6 |          1.4 |         0.2 |
user> (ds/head (cf/categorical ds))
https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv [5 1]:

| species |
|---------|
|  setosa |
|  setosa |
|  setosa |
|  setosa |
|  setosa |
user> (def numeric-ds (ds/categorical->number ds cf/categorical))
#'user/numeric-ds
user> (ds/head numeric-ds)
https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv [5 5]:

| sepal_length | sepal_width | petal_length | petal_width | species |
|--------------|-------------|--------------|-------------|---------|
|          5.1 |         3.5 |          1.4 |         0.2 |     1.0 |
|          4.9 |         3.0 |          1.4 |         0.2 |     1.0 |
|          4.7 |         3.2 |          1.3 |         0.2 |     1.0 |
|          4.6 |         3.1 |          1.5 |         0.2 |     1.0 |
|          5.0 |         3.6 |          1.4 |         0.2 |     1.0 |
user> (meta (numeric-ds "species"))
{:categorical? true,
 :name "species",
 :datatype :float64,
 :n-elems 150,
 :categorical-map
 {:lookup-table {"versicolor" 0, "setosa" 1, "virginica" 2},
  :src-column "species",
  :result-datatype :float64}}

;;More transforms like this including one-hot and inverting the mapping are
;;available in tech.v3.dataset.categorical.
```

### Regression

For regression we will mark the `petal_width` as the field we want to infer on and
use the default xgboost regression model.  Moving into actual modelling, we will
including the `tech.v3.dataset.modelling` namespace and the xgboost bindings.


We will use a method were we split the dataset into train/test datasets
via random sampling, train the model, and calculate the loss using the
test-ds and the default regression loss function - mean average error or
`mae`:


```clojure
user> (require '[tech.v3.dataset.modelling :as ds-mod])
nil
user> (def regression-ds (ds-mod/set-inference-target numeric-ds "petal_width"))
#'user/regression-ds
user> (require '[tech.v3.libs.xgboost])
nil
;; Also tech.v3.libs.smile.regression and tech.v3.libs.smile.classification provide quite
;; a few models.
user> (require '[tech.v3.ml :as ml])
nil
user> (def model (ml/train-split regression-ds {:model-type :xgboost/regression}))
#'user/model
user> (:loss model)
0.1272654171784719
```

We split the dataset into k datasets via the k-fold algorithm and get more error information:

```clojure
user> (def k-fold-model (ml/train-k-fold regression-ds {:model-type :xgboost/regression}))
#'user/k-fold-model
user> (select-keys k-fold-model [:min-loss :avg-loss :max-loss])
{:min-loss 0.1050250555238416,
 :avg-loss 0.1569393319631037,
 :max-loss 0.19597491553196542}
```

Given a model we can predict what the answer will be for the column the model
was trained for:

```clojure
user> (ds/head (ml/predict regression-ds k-fold-model))
:_unnamed [5 1]:

| petal_width |
|-------------|
|      0.2615 |
|      0.1569 |
|      0.1862 |
|      0.1780 |
|      0.2410 |
```

And thus calculating our own loss is easy:

```clojure
user> (require '[tech.v3.ml.loss :as loss])
nil
user> (def predictions (ml/predict regression-ds k-fold-model))
#'user/predictions
user> (loss/mae (predictions "petal_width")
                (regression-ds "petal_width"))
0.04563391995429992
```

The loss in this case is artificially low because we are testing on the same
data that we trained on.


### Classification

For classification will we attempt to predict the species.


```clojure
user> (def classification-ds (ds-mod/set-inference-target numeric-ds "species"))
#'user/classification-ds
user> (def k-fold-model (ml/train-k-fold classification-ds {:model-type :xgboost/classification}))
#'user/k-fold-model
user> (select-keys k-fold-model [:min-loss :avg-loss :max-loss])
{:min-loss 0.0, :avg-loss 0.04516129032258065, :max-loss 0.09677419354838712}
```

The XGBoost system has a powerful classification engine!


We can ask the model which columns it found the most useful:

```clojure
user> (ml/explain k-fold-model)
_unnamed [4 3]:

| :importance-type |     :colname |      :gain |
|------------------|--------------|------------|
|             gain | petal_length | 3.38923719 |
|             gain |  petal_width | 2.50506807 |
|             gain | sepal_length | 0.22811251 |
|             gain |  sepal_width | 0.22783045 |
```


When we predict a classification dataset we get back a probability distribution along with
the original column mapped to whatever index had the max probability:


```clojure
user> (ds/head (ml/predict classification-ds k-fold-model))
:_unnamed [5 4]:

| versicolor | setosa | virginica | species |
|------------|--------|-----------|---------|
|   0.006784 | 0.9902 |  0.003023 |     1.0 |
|   0.006032 | 0.9900 |  0.003945 |     1.0 |
|   0.006348 | 0.9906 |  0.003025 |     1.0 |
|   0.006348 | 0.9906 |  0.003031 |     1.0 |
|   0.006348 | 0.9906 |  0.003025 |     1.0 |

;;The columns are marked with their type:

user> (map meta (vals *1))
({:name "versicolor",
  :datatype :object,
  :n-elems 5,
  :column-type :probability-distribution}
 {:name "setosa",
  :datatype :object,
  :n-elems 5,
  :column-type :probability-distribution}
 {:name "virginica",
  :datatype :object,
  :n-elems 5,
  :column-type :probability-distribution}
 {:categorical? true,
  :categorical-map
  {:lookup-table {"versicolor" 0, "setosa" 1, "virginica" 2},
   :src-column "species",
   :result-datatype :float64},
  :name "species",
  :datatype :float64,
  :n-elems 5,
  :column-type :prediction})
```

Due to the metadata saved on the `species` column we can reverse map back to the
original column values using the `tech.v3.dataset.categorical` namespace:


```clojure
user> (require '[tech.v3.dataset.categorical :as ds-cat])
nil
user> (ds/head (ds-cat/reverse-map-categorical-xforms predictions))
:_unnamed [5 4]:

| versicolor | setosa | virginica | species |
|------------|--------|-----------|---------|
|   0.006784 | 0.9902 |  0.003023 |  setosa |
|   0.006032 | 0.9900 |  0.003945 |  setosa |
|   0.006348 | 0.9906 |  0.003025 |  setosa |
|   0.006348 | 0.9906 |  0.003031 |  setosa |
|   0.006348 | 0.9906 |  0.003025 |  setosa |
```

### Concluding


We have generic support for xgboost and smile.  This gives you quite a few models and
they are all gridsearcheable.  We put this forward in an attempt to simplify
doing ML that we do and in an attempt to move the Clojure ML conversation forward
towards getting the best possible results for a dataset in the least amount of
(developer) time.



* For a more in-depth walkthrough of XGBoost features, checkout the
  [XGBoost topic](topics/xgboost_metrics.md).

## License

Copyright © 2019 Tech Ascent, LLC

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
