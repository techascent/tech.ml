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
user> (require '[tech.v2.datatype.functional :as dfn])
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
user> (require '[tech.ml :as ml])
nil
user> (require '[tech.libs.xgboost])
nil
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

## Gridsearching

We can gridsearch through the xgboost options in order to find the 'best' options
for a dataset.

We first build out an option map where some of the keys map to gridsearch commands.
The xgboost model can fill out gridsearch options:

```clojure
user> (def model-options {:model-type :xgboost/regression})
#'user/model-options
user> (ml/auto-gridsearch-options model-options)
{:subsample #function[tech.ml.gridsearch/make-gridsearch-fn/fn--44403],
 :scale-pos-weight #function[tech.ml.gridsearch/make-gridsearch-fn/fn--44403],
 :max-depth #function[clojure.core/comp/fn--5807],
 :lambda #function[tech.ml.gridsearch/make-gridsearch-fn/fn--44403],
 :gamma #function[tech.ml.gridsearch/make-gridsearch-fn/fn--44403],
 :eta #function[tech.ml.gridsearch/make-gridsearch-fn/fn--44403],
 :alpha #function[tech.ml.gridsearch/make-gridsearch-fn/fn--44403],
 :model-type :xgboost/regression}
```

Once we have a map where some of the keys map to gridsearch entries, we can use the
automatic gridsearch facility in tech.ml to search over the space:

```clojure
user> (def gridsearch (ml/gridsearch (ml/auto-gridsearch-options
                                      {:model-type :xgboost/regression})
                                     loss/mae (:train-ds train-test-split)))
07:49:40.244 [nRepl-session-1b8515ca-e7bf-4cae-9f3d-310d1c86239e] INFO  tech.ml - Gridsearching: {:top-n 5, :gridsearch-depth 50, :k-fold 5}

#'user/gridsearch

user> (count gridsearch)
5
user> (keys (first gridsearch))
(:total-train-time
 :predict-time
 :id
 :options
 :loss
 :train-time
 :total-predict-time
 :average-loss
 :model)
user> (map :average-loss gridsearch)
(17689.661645983007
 18249.499280650056
 18354.93181460924
 19183.83188931946
 19687.72537490944)
```

Note that gridsearching saves out the option map so we can see what produced the best
options or perform a new sub-gridsearch given ranges built from the return value
of the previous gridsearch.  We remove keys that are added into the options by the
dataset system to get a cleaner map for presentation:

```clojure

user> (dissoc (:options (first gridsearch))
              :feature-columns
              :label-columns
              :label-map
              :column-map
              :dataset-shape)
{:subsample 0.690625,
 :scale-pos-weight 1.346875,
 :lambda 0.6940625,
 :model-type :xgboost/regression,
 :gamma 0.052329911468149484,
 :alpha 0.11984165845261181,
 :max-depth 17,
 :eta 0.15625}
```

Doing this over the return value of gridsearch is instructive.  You can see what
the various options mean
[here](https://xgboost.readthedocs.io/en/latest/parameter.html).  Regardless, we
will just use the best options for now:

```clojure
user> (def model-options (dissoc (:options (first gridsearch))
                                 :feature-columns
                                 :label-columns
                                 :label-map
                                 :column-map
                                 :dataset-shape))
#'user/model-options
```

## Metrics

Metrics will help us see if the model itself is overtraining.  When you setup
xgboost options with one or more `watch` datasets, it will dump out the metrics
generated during training to a map of the same name under the model:

```clojure
user> (def model (ml/train (assoc model-options
                                  :watches {:test-ds (:test-ds train-test-split)}
                                  :eval-metric "mae")
                           (:train-ds train-test-split)))
#'user/model
user> (get-in model [:model :metrics :test-ds])
[155104.42, 131292.08, 111617.01, 94516.586, 80522.01, 68706.055, 58783.195,
 50716.758, 43895.25, 38271.21, 33724.87, 30095.79, 27317.229, 25235.572,
 23583.646, 22291.254, 21304.0, 20369.557, 19870.934, 19353.86, 19033.955,
 18718.123, 18505.295, 18273.787, 18161.504]
user> (loss/mae (ml/predict model (:test-ds train-test-split))
                (ds/labels (:test-ds train-test-split)))
18161.504842679795
```

Our loss lines up with our metrics.  In this case it appears the default number of
training rounds, 25, works pretty well.  That is luck and dataset dependent.  For
instance, we could just as easily chosen to train more rounds:


```clojure
user> (def model (ml/train (assoc model-options
                                  :watches {:test-ds (:test-ds train-test-split)}
                                  :eval-metric "mae"
                                  :round 50)
                           (:train-ds train-test-split)))
#'user/model
user> (get-in model [:model :metrics :test-ds])
[155104.42, 131292.08, 111617.016, 94516.59, 80522.01, 68706.06, 58783.195,
 50716.758, 43895.254, 38271.21, 33724.87, 30095.79, 27317.227, 25235.572,
 23583.648, 22291.254, 21304.0, 20369.557, 19870.934, 19353.86, 19033.957,
 18718.125, 18505.295, 18273.787, 18161.506, 18063.521, 17976.748, 17840.676,
 17758.293, 17679.395, 17644.885, 17611.982, 17594.037, 17564.527, 17559.14,
 17526.754, 17505.686, 17478.936, 17479.887, 17460.365, 17476.04, 17463.885,
 17430.012, 17414.898, 17416.896, 17407.037, 17404.498, 17401.482, 17402.996,
 17390.938]
```

Now we see a very common case.  XGBoost is overtraining; the error on the validation
set is going up while the model continues to train further.  We can make this
clearer by add in the training dataset to the watches map:

```clojure
user> (def model (ml/train (assoc model-options
                                  :watches {:test-ds (:test-ds train-test-split)
                                            :train-ds (:train-ds train-test-split)}
                                  :eval-metric "mae"
                                  :round 100)
                           (:train-ds train-test-split)))
#'user/model
user> (-> (ds/name-values-seq->dataset (get-in model [:model :metrics]))
          (ds/select :all (range 80 100)))
_unnamed [20 2]:

|  :test-ds | :train-ds |
|-----------+-----------|
| 17354.963 |   130.350 |
| 17352.686 |   122.937 |
| 17351.160 |   117.777 |
| 17350.686 |   109.455 |
| 17350.396 |   103.957 |
| 17351.082 |    99.027 |
| 17351.402 |    93.890 |
| 17351.469 |    88.454 |
| 17352.172 |    83.471 |
...
```

I selected the last 20 rows as those are the ones that show the overtraining somewhat.


## Early Stopping


Using the built-in XGBoost early stopping we can avoid overtraining:


```clojure

user> (def model (ml/train (assoc model-options
                                  :watches {:test-ds (:test-ds train-test-split)
                                            :train-ds (:train-ds train-test-split)}
                                  :eval-metric "mae"
                                  :round 100
                                  :early-stopping-round 4)
                           (:train-ds train-test-split)))
08:06:07.652 [nRepl-session-1b8515ca-e7bf-4cae-9f3d-310d1c86239e] WARN  tech.libs.xgboost - Early stopping indicated but watches has undefined iteration order.
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

user> (def model (ml/train (assoc model-options
                                  :watches watches
                                  :eval-metric "mae"
                                  :round 100
                                  :early-stopping-round 4)
                           (:train-ds train-test-split)))
#'user/model
user> (def metric-ds (-> (ds/name-values-seq->dataset (get-in model [:model :metrics]))
                         (ds/add-or-update-column "Round" (int-array (range 100)))))
#'user/metric-ds
user> (def filtered-metrics (ds/ds-filter #(> (get % :test-ds) 0.0) metric-ds))
#'user/filtered-metrics
user> filtered-metrics
_unnamed [61 3]:

|  :train-ds |   :test-ds | Round |
|------------+------------+-------|
| 152023.609 | 155104.422 |     0 |
| 128871.703 | 131292.078 |     1 |
| 109361.906 | 111617.023 |     2 |
|  92792.797 |  94516.602 |     3 |
|  78915.680 |  80522.008 |     4 |
|  66949.461 |  68706.063 |     5 |
|  56878.953 |  58783.195 |     6 |
|  48459.984 |  50716.762 |     7 |
|  41276.801 |  43895.254 |     8 |
|  35206.266 |  38271.211 |     9 |
|  30170.004 |  33724.871 |    10 |
|  25856.465 |  30095.787 |    11 |
|  22220.389 |  27317.229 |    12 |
...

user> (require '[tech.v2.datatype :as dtype])
nil
user> (ds/select filtered-metrics
                 :all
                 (take-last 10 (range (second (dtype/shape filtered-metrics)))))
_unnamed [10 3]:

| :train-ds |  :test-ds | Round |
|-----------+-----------+-------|
|   741.896 | 17386.498 |    51 |
|   688.998 | 17372.967 |    52 |
|   664.834 | 17369.799 |    53 |
|   629.150 | 17361.590 |    54 |
|   597.003 | 17352.551 |    55 |
|   562.561 | 17351.961 |    56 |
|   534.878 | 17361.086 |    57 |
|   507.697 | 17360.170 |    58 |
|   473.220 | 17360.492 |    59 |
|   451.281 | 17359.107 |    60 |
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
