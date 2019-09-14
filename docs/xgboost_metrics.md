# XGBoost Metrics & Early Stopping


Recently we upgraded access to the xgboost machine learning system to include
metrics and early stopping.  This document will be a quick walkthough using the ames
dataset to show how to use both systems.


## Dataset Processing


Our goal is to end up with float64 columns with no missing values.  This is a fast
rough pass and isn't anywhere near ideal.  For instance many of the string columns
should be encoded to preserve semantic order.  For a more thorough treatment
of this dataset please see our [ames tutorial](https://github.com/cnuernber/ames-house-prices/blob/master/ames-housing-prices-clojure.md).

```clojure

user> (require '[tech.ml.dataset :as ds])
nil
user> (require '[tech.ml.dataset.column :as ds-col])
nil
user> (def ames-ds (ds/->dataset "../tech.ml.dataset/data/ames-house-prices/train.csv.gz"))
#'user/ames-ds
user> (ds/columns-with-missing-seq ames-ds)
({:column-name "LotFrontage", :missing-count 259}
 {:column-name "Alley", :missing-count 1369}
 {:column-name "MasVnrType", :missing-count 8}
 {:column-name "MasVnrArea", :missing-count 8}
 {:column-name "BsmtQual", :missing-count 37}
 {:column-name "BsmtCond", :missing-count 37}
 {:column-name "BsmtExposure", :missing-count 38}
 {:column-name "BsmtFinType1", :missing-count 37}
 {:column-name "BsmtFinType2", :missing-count 38}
 {:column-name "Electrical", :missing-count 1}
 {:column-name "FireplaceQu", :missing-count 690}
 {:column-name "GarageType", :missing-count 81}
 {:column-name "GarageYrBlt", :missing-count 81}
 {:column-name "GarageFinish", :missing-count 81}
 {:column-name "GarageQual", :missing-count 81}
 {:column-name "GarageCond", :missing-count 81}
 {:column-name "PoolQC", :missing-count 1453}
 {:column-name "Fence", :missing-count 1179}
 {:column-name "MiscFeature", :missing-count 1406})
user> (->> ames-ds
           (map dtype/get-datatype)
           frequencies)
{:int32 38, :string 42, :boolean 1}
user> (require '[tech.ml.dataset.pipeline :as ds-pipe])
nil
user> (require '[tech.ml.dataset.pipeline.column-filters :as col-filters])
nil
user> (col-filters/missing? ames-ds)
("Id"
 "MSSubClass"
 "MSZoning"
 "LotFrontage"
 "LotArea"
 "Street"
 ...)
user> (def ames-processed
        (-> ames-ds
            (ds-pipe/string->number)
            (ds-pipe/->datatype)
			(ds-pipe/replace-missing col-filters/missing? dfn/mean)
            (ds/remove-column "Id")
            (ds/set-inference-target "SalePrice")))
#'user/ames-processed
user> (->> ames-processed
           (map dtype/get-datatype)
           frequencies)
{:float64 80}
user> (ds/columns-with-missing-seq ames-processed)
nil
```

## Training with XGBoost


Since we set the inference target on the dataset, we can quickly train a model.
```clojure
user> (def model (ml/train {:model-type :xgboost/regression} ames-processed))
#'user/model
```

But since we have no validation set we can't really say how good it is.  So we
split the dataset up into train/test datasets where we can train on one dataset
and test on another.


```clojure
user> (def train-test-split (ds/->train-test-split ames-processed))
#'user/train-test-split
user> (def model (ml/train {:model-type :xgboost/regression}
                           (:train-ds train-test-split)))
#'user/model
```

Now we can say something about how good the model is.  Let's analyze this with
mean average error:

```clojure
user> (require '[tech.ml.loss :as loss])
nil
user> (loss/mae (ml/predict model (:test-ds train-test-split))
                (ds/labels (:test-ds train-test-split)))

18214.909737086185
```

What is going on?  Well, one question we want to to ask is what variables is xgboost
using to decide how to predict SalePrice.

```clojure
user> (ml/explain-model model)
{"gain"
 (["OverallQual" 2.5181549976236365E11]
  ["GarageCars" 1.4100606415566666E11]
  ["BsmtQual" 3.9808249456E10]
  ["GrLivArea" 3.41621977572E10]
  ["KitchenQual" 2.4555688343714287E10]
  ["TotRmsAbvGrd" 2.12383297232E10]
  ["ExterQual" 1.2918698357333334E10]
  ["TotalBsmtSF" 1.1815750909642857E10]
  ["KitchenAbvGr" 9.67573504E9]
  ["1stFlrSF" 9.35139666304E9]
  ["BsmtFinSF1" 6.262918230857142E9]
  ...)}
```

These seem logical.  In fact, it would appear that these columns are in this case
somewhat correlated with the pearson correlation table for sale price:

```clojure
(ds/correlation-table ames-processed :colname-seq ["SalePrice"])
{"SalePrice"
 (["SalePrice" 1.0]
  ["OverallQual" 0.7909816005838052]
  ["GrLivArea" 0.7086244776126517]
  ["ExterQual" 0.6501302285588267]
  ["GarageCars" 0.6404091972583521]
  ["GarageArea" 0.6234314389183621]
  ["KitchenQual" 0.6192349321077227]
  ["TotalBsmtSF" 0.6135805515591942]
  ["BsmtQual" 0.6104442034754758]
  ["1stFlrSF" 0.6058521846919152]
  ["FullBath" 0.5606637627484452]
  ["TotRmsAbvGrd" 0.5337231555820283]
  ...)}
```


## Metrics

Metrics will help us see if the model itself is overtraining.  When you setup
xgboost options with one or more `watch` datasets, it will dump out the metrics
generated during training to a map of the same name under the model:

```clojure
user> (def model (ml/train {:model-type :xgboost/regression
                            :watches {:test-ds (:test-ds train-test-split)}
                            :eval-metric "mae"}
                           (:train-ds train-test-split)))
#'user/model
user> (get-in model [:model :metrics :test-ds])
[127331.15, 90284.18, 64315.516, 46692.58, 35922.11, 28256.158, 23941.064, 21352.5,
 19771.996, 18930.922, 18512.373, 18119.22, 17872.537, 17753.398, 17720.832,
 17666.295, 17599.432, 17616.467, 17598.205, 17585.389, 17567.115, 17571.035,
 17634.732, 17655.084, 17625.824]

user> (loss/mae (ml/predict model (:test-ds train-test-split))
                (ds/labels (:test-ds train-test-split)))
18214.909737086185
```

Our loss lines up with our metrics.  In this case it appears the default number of
training rounds, 25, works pretty well.  That is luck and dataset dependent.  For
instance, we could just as easily chosen to train more rounds:


```clojure
(def model (ml/train {:model-type :xgboost/regression
                            :watches {:test-ds (:test-ds train-test-split)}
                            :eval-metric "mae"
                            :round 50}
                           (:train-ds train-test-split)))
#'user/model
user> (get-in model [:model :metrics :test-ds])
[127331.14, 90284.17, 64315.508, 46692.586, 35922.113, 28256.154, 23941.066,
 21352.502, 19771.996, 18930.922, 18512.375, 18119.22, 17872.535, 17753.4,
 17720.834, 17666.295, 17599.43, 17616.469, 17598.207, 17585.389, 17567.117,
 17571.033, 17634.732, 17655.084, 17625.822, 17625.268, 17574.975, 17502.158,
 17518.38, 17448.934, 17457.328, 17483.254, 17478.21, 17411.47, 17422.34, 17380.65,
 17385.346, 17360.656, 17373.139, 17353.418, 17312.076, 17322.57, 17300.586,
 17286.68, 17266.291, 17269.02, 17250.287, 17289.46, 17287.64, 17316.69]
```

Now we see a very common case.  XGBoost is overtraining; the error on the validation
set is going up while the model continues to train further.  We can make this
clearer by add in the training dataset to the watches map:

```clojure

user> (def model (ml/train {:model-type :xgboost/regression
                            :watches {:test-ds (:test-ds train-test-split)
                                      :train-ds (:train-ds train-test-split)}
                            :eval-metric "mae"
                            :round 100}
                           (:train-ds train-test-split)))
#'user/model
user> (-> (ds/name-values-seq->dataset (get-in model [:model :metrics]))
          (ds/select :all (range 80 100)))
_unnamed [20 2]:

|  :test-ds | :train-ds |
|-----------+-----------|
| 17302.588 |   793.363 |
| 17299.600 |   773.398 |
| 17285.396 |   735.530 |
| 17289.949 |   707.300 |
| 17292.154 |   684.866 |
| 17292.359 |   676.433 |
| 17294.512 |   659.553 |
...
```

I selected the last 20 rows as those are the ones that show the overtraining somewhat.


## Early Stopping


Using the built-in XGBoost early stopping we can avoid overtraining:


```clojure
user> (def model (ml/train {:model-type :xgboost/regression
                            :watches {:test-ds (:test-ds train-test-split)
                                      :train-ds (:train-ds train-test-split)}
                            :eval-metric "mae"
                            :round 100
                            :early-stopping-round 4}
                           (:train-ds train-test-split)))
14:26:11.537 [nRepl-session-b108830c-450a-454f-9885-50f4b17d1d53] WARN  tech.libs.xgboost - Early stopping indicated but watches has undefined iteration order.
Early stopping will always use the 'last' of the watches as defined by the iteration
order of the watches map.  Consider using a java.util.LinkedHashMap for watches.
https://github.com/dmlc/xgboost/blob/master/jvm-packages/xgboost4j/src/main/java/ml/dml
c/xgboost4j/java/XGBoost.java#L208
#'user/model
```

Oops!  This is a implementation detail of xgboost.  We have to use a map that
retains insertion order in order to do early stopping or we have to have only
one watch.


```clojure

user> (import '[java.util LinkedHashMap])
java.util.LinkedHashMap
user> (def watches (doto (LinkedHashMap.)
                     (.put :train-ds (:train-ds train-test-split))
                     (.put :test-ds (:test-ds train-test-split))))
#'user/watches
user> (def model (ml/train {:model-type :xgboost/regression
                            :watches watches
                            :eval-metric "mae"
                            :round 100
                            :early-stopping-round 4}
                           (:train-ds train-test-split)))
#'user/model
user> (def metric-ds (-> (ds/name-values-seq->dataset (get-in model [:model :metrics]))
                         (ds/add-or-update-column "Round" (int-array (range 100)))))
#'user/metric-ds
user> (def filtered-metrics (ds/ds-filter #(> (get % :test-ds) 0.0) metric-ds))
#'user/filtered-metrics
filtered-metrics
_unnamed [25 3]:

|  :train-ds |   :test-ds | Round |
|------------+------------+-------|
| 127692.180 | 127331.141 |     0 |
|  90460.734 |  90284.180 |     1 |
|  64045.789 |  64315.516 |     2 |
|  45496.582 |  46692.578 |     3 |
|  32755.637 |  35922.109 |     4 |
|  23874.713 |  28256.154 |     5 |
|  18172.201 |  23941.064 |     6 |
|  14431.069 |  21352.500 |     7 |
...

user> (require '[tech.v2.datatype.functional :as dfn])
nil

;;We made it to round 25--
user> (dfn/reduce-max (filtered-metrics "Round"))
24
```

So here we asked XGBoost to stop if the error on the evaluation dataset (the 'last'
dataset in the watches map) rises 4 rounds consecutively.


## Caveat - Cleaning the Options Map


The options used to train the model are stored verbatim in the model map.  This means
that if you use watches and then save the model you will probably get an error.
The best way around this is to remove watches from the model option map before
you attemp to nippy the data:


```clojure

user> (keys (:options model))
(:dataset-shape
 :feature-columns
 :watches
 :label-columns
 :round
 :early-stopping-round
 :model-type
 :eval-metric
 :label-map
 :column-map)
user> (def clean-model (update model :options dissoc :watches))
#'user/clean-model
user> (keys (:options clean-model))
(:dataset-shape
 :feature-columns
 :label-columns
 :round
 :early-stopping-round
 :model-type
 :eval-metric
 :label-map
 :column-map)
```


## Conclusion


We grabbed a dataset, made it trainable (no missing, no strings, everything is
double datatype).  We then trained an initial model and checked out which columns
XGBoost found useful.  We furthermore built out metrics using xgboost's watches
feature and observed error rates as XGBoost continued to train.  We then ask xgboost
to stop training if the error on a validation dataset (the last of the watches map
by iteration) increased 4 rounds consecutively.
