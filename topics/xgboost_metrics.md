# XGBoost Metrics & Early Stopping


Recently we upgraded access to the xgboost machine learning system to include
metrics and early stopping.  This document will be a quick walkthough using the Ames
housing prices dataset to show how to use both systems.


## Dataset Processing


Our goal is to end up with float64 columns with no missing values.  This is a fast
rough pass and isn't anywhere near ideal.  For instance many of the string columns
should be encoded to preserve semantic order.  Regardless this will show some minimal
processing and the general pathway in order to do simple machine learning.

```clojure

user> (require '[tech.v3.dataset :as ds])
nil
user> (def ames (ds/->dataset "https://github.com/techascent/tech.ml/blob/dataset-5.X/test/data/train.csv.gz?raw=true"
                              {:file-type :csv :gzipped? true}))
#'user/ames
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
user> (->> (map (comp :datatype meta) (vals ames))
           (frequencies))
{:int16 36, :string 42, :int32 2, :boolean 1}
user> (require '[tech.v3.dataset.column-filters :as cf])
nil
user> (def ames-missing-1 (ds/replace-missing-value ames cf/string "NA"))
#'user/ames-missing-1
user> (ds/columns-with-missing-seq ames-missing-1)
({:column-name "LotFrontage", :missing-count 259}
 {:column-name "MasVnrArea", :missing-count 8}
 {:column-name "GarageYrBlt", :missing-count 81})
user> (def ames-missing-2 (ds/replace-missing-value ames-missing-1 :all 0))
#'user/ames-missing-2
user> (ds/columns-with-missing-seq ames-missing-2)
nil
user> (def all-numeric (ds/categorical->number ames-missing-2 cf/string))
#'user/all-numeric
user> (->> (map (comp :datatype meta) (vals all-numeric))
           (frequencies))
{:int16 36, :float64 42, :int32 2, :boolean 1}
```

## Training with XGBoost


Since we set the inference target on the dataset, we can quickly train a model.
```clojure
user> (require '[tech.v3.ml :as ml])
nil
user> (require '[tech.v3.libs.xgboost])
nil
user> (require '[tech.v3.dataset.modelling :as ds-mod])
nil
```

We split the dataset up into train/test datasets where we can train on one dataset
and test on another.


```clojure
user> (def train-test-split (ds/->train-test-split ames-processed))
#'user/train-test-split
user> (def model (ml/train {:model-type :xgboost/regression}
                           (:train-ds train-test-split)))
#'user/model
user> (def prediction (ml/predict (:test-ds train-test-split) model))
#'user/prediction
user> prediction
:_unnamed [438 1]:

| SalePrice |
|-----------|
| 1.904E+05 |
| 2.000E+05 |
| 1.944E+05 |
| 2.392E+05 |
| 1.766E+05 |
...
user> (meta prediction)
{:name :_unnamed, :model-type :regression}
```

Now we can say something about how good the model is.  Let's analyze this with
mean average error:

```clojure
user> (require '[tech.v3.ml.loss :as loss])
nil
user> (loss/mae (prediction "SalePrice")
                ((:test-ds train-test-split) "SalePrice"))
18439.617499643264
```

What is going on?  Well, one question we want to to ask is what variables is xgboost
using to decide how to predict SalePrice.

```clojure

user> (ml/explain model)
_unnamed [70 3]:
| :importance-type |     :colname |          :gain |
|------------------|--------------|----------------|
|             gain |  OverallQual | 2.63303470E+11 |
|             gain |  KitchenQual | 5.15596265E+10 |
|             gain | TotRmsAbvGrd | 4.91487037E+10 |
|             gain |    GrLivArea | 3.93336045E+10 |
|             gain |   GarageCars | 3.66395853E+10 |
|             gain |   BsmtFinSF1 | 1.28130489E+10 |
|             gain |  TotalBsmtSF | 1.04417586E+10 |
|             gain |     BsmtQual | 8.16779754E+09 |
|             gain |     MSZoning | 7.97729807E+09 |
|             gain |     1stFlrSF | 6.55107315E+09 |
|             gain | BsmtExposure | 6.07410704E+09 |
|             gain |     2ndFlrSF | 5.68743133E+09 |
|             gain |        Alley | 5.56387533E+09 |
|             gain |   Fireplaces | 4.28944688E+09 |
|             gain |    YearBuilt | 3.90289328E+09 |
|             gain |  LotFrontage | 3.21221797E+09 |
|             gain | GarageFinish | 3.10737368E+09 |
|             gain | YearRemodAdd | 3.08412712E+09 |
|             gain |  GarageYrBlt | 3.01600398E+09 |
|             gain |     SaleType | 2.76226739E+09 |
|             gain |    LandSlope | 2.70233190E+09 |
|             gain |      LotArea | 2.42620203E+09 |
|             gain | BsmtFinType2 | 2.29969101E+09 |
|             gain |   Foundation | 2.27973530E+09 |
|             gain | BsmtFullBath | 2.17456462E+09 |
```

These seem logical.  In fact, it would appear that these columns are in this case
somewhat correlated with the pearson correlation table for sale price:

```clojure
user> (require '[tech.v3.dataset.math :as ds-math])
nil
user> (ds-math/correlation-table all-numeric :colname-seq ["SalePrice"])
WARNING - excluding non-numeric columns:
 [CentralAir]
{"SalePrice"
 (["SalePrice" 1.0]
  ["OverallQual" 0.7909816005838052]
  ["GrLivArea" 0.7086244776126517]
  ["ExterQual" 0.6501302285588267]
  ["GarageCars" 0.6404091972583521]
  ["GarageArea" 0.6234314389183621]
  ["KitchenQual" 0.6192349321077227]
  ...)}
```

## Gridsearching

We can gridsearch through the xgboost options in order to find the 'best' options
for a dataset.

We first build out an option map where some of the keys map to gridsearch commands.
The xgboost model can fill out gridsearch options:

```clojure
user> (ml/hyperparameters :xgboost/regression)
{:subsample
 {:tech.v3.ml.gridsearch/type :linear,
  :start 0.7,
  :end 1.0,
  :n-steps 3,
  :result-space :float64},
 :scale-pos-weight
 {:tech.v3.ml.gridsearch/type :linear,
  :start 0.7,
  :end 1.31,
  :n-steps 6,
  :result-space :float64},
 :max-depth
 {:tech.v3.ml.gridsearch/type :linear,
  :start 1.0,
  :end 10.0,
  :n-steps 10,
  :result-space :int64},
```

Once we have a map where some of the keys map to gridsearch entries, we can use the
automatic gridsearch facility in tech.ml to search over the space:

```clojure
user> (def gridsearchable-options (merge {:model-type :xgboost/regression} (ml/hyperparameters :xgboost/regression)))
#'user/gridsearchable-options
user> (def option-seq (take 100 (gs/sobol-gridsearch gridsearchable-options)))
#'user/option-seq
user> (take 5 option-seq)
({:subsample 0.85,
  :scale-pos-weight 1.066,
  :lambda 0.16517241379310346,
  :round 25,
  :model-type :xgboost/regression,
  :gamma 0.556,
  :alpha 0.16517241379310346,
  :max-depth 6,
  :eta 0.5555555555555556}
 {:subsample 1.0,
  :scale-pos-weight 0.822,
  :lambda 0.08241379310344828,
  :round 15,
  :model-type :xgboost/regression,
  :gamma 0.778,
  :alpha 0.23758620689655172,
  :max-depth 3,
  :eta 0.7777777777777778}
...
user> (defn test-options
        [options]
        (let [model (ml/train (:train-ds train-test-split) options)
              prediction (ml/predict (:test-ds train-test-split) model)]
          (assoc model :loss (loss/mae (prediction "SalePrice")
                                       ((:test-ds train-test-split) "SalePrice")))))

#'user/test-options
user> (def search-results
        (->> option-seq
             (map test-options)
             (sort-by :loss)
             (take 10)))
#'user/search-results
user> (map :loss search-results)
(17407.003585188355
 17640.71050049943
 17647.76534853025
 17656.68644763128
 17748.619631135844
 17873.145191210046
 17911.20817280251
 18073.004726740866
 18221.194652539954
 18305.031258918378)
```

Note that gridsearching saves out the option map so we can see what produced the best
options or perform a new sub-gridsearch given ranges built from the return value
of the previous gridsearch.

```clojure
user> (:options (first gridsearch))
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
user> (def model-options (:options (first search-results)))
#'user/model-options
```

## Metrics

Metrics will help us see if the model itself is overtraining.  When you setup
xgboost options with one or more `watch` datasets, it will dump out the metrics
generated during training to a map of the same name under the [:model-data :metrics]
path:

```clojure
user> (def model (ml/train (:train-ds train-test-split)
                           (assoc model-options
                                  :watches {:test-ds (:test-ds train-test-split)}
                                  :eval-metric "mae")))
#'user/model
user> (get-in model [:model :metrics])
nil
user> (get-in model [:model-data :metrics])
:metrics [35 1]:

|        :test-ds |
|-----------------|
| 164215.03125000 |
| 145998.57812500 |
| 129672.75000000 |
| 115247.78906250 |
| 102420.60156250 |
...
user> (ds/tail (get-in model [:model-data :metrics]))
:metrics [5 1]:

|       :test-ds |
|----------------|
| 17594.75390625 |
| 17569.87109375 |
| 17500.19921875 |
| 17438.49609375 |
| 17407.00390625 |
user> (loss/mae ((ml/predict (:test-ds train-test-split) model) "SalePrice")
                ((:test-ds train-test-split) "SalePrice"))
17407.003585188355
```

Our loss lines up with our metrics.  In this case it appears the default number of
training rounds, 25, works pretty well.  That is luck and dataset dependent.  For
instance, we could just as easily chosen to train more rounds:


```clojure
user> (def model (ml/train (:train-ds train-test-split)
                           (assoc model-options
                                  :watches {:test-ds (:test-ds train-test-split)}
                                  :eval-metric "mae"
                                  :round 100)))
user> (ds/tail (get-in model [:model-data :metrics]) 20)
:metrics [20 1]:

|       :test-ds |
|----------------|
| 17170.25976563 |
| 17169.70117188 |
| 17169.28906250 |
| 17170.50585938 |
| 17170.07226563 |
| 17168.27929688 |
| 17168.89062500 |
| 17173.72851563 |
| 17179.31640625 |
| 17177.67773438 |
| 17175.00000000 |
| 17172.91406250 |
| 17170.39257813 |
| 17166.40039063 |
| 17166.11132813 |
| 17162.64648438 |
| 17161.11328125 |
| 17161.10351563 |
| 17157.91796875 |
| 17159.54492188 |
```

Now we see a very common case.  XGBoost is overtraining; the error on the validation
set not improving while the model continues to train further.  We can make this
clearer by adding in the training dataset to the watches map:

```clojure
user> (def model (ml/train (:train-ds train-test-split)
                           (assoc model-options
                                  :watches train-test-split
                                  :eval-metric "mae"
                                  :round 100)))
#'user/model
user> (ds/tail (get-in model [:model-data :metrics]) 20)
:metrics [20 2]:

|    :train-ds |       :test-ds |
|--------------|----------------|
| 833.36267090 | 17170.25781250 |
| 802.52020264 | 17169.70117188 |
| 777.55834961 | 17169.28906250 |
| 742.71563721 | 17170.50585938 |
| 712.96319580 | 17170.07226563 |
| 697.16204834 | 17168.27734375 |
| 668.31335449 | 17168.89062500 |
| 636.66845703 | 17173.72851563 |
| 606.96020508 | 17179.31640625 |
| 583.48291016 | 17177.67773438 |
| 564.89141846 | 17175.00000000 |
| 542.37860107 | 17172.91406250 |
| 529.37158203 | 17170.39257813 |
| 515.00244141 | 17166.40039063 |
| 494.43447876 | 17166.11132813 |
| 473.96432495 | 17162.64648438 |
| 456.32290649 | 17161.11328125 |
| 448.66156006 | 17161.10351563 |
| 439.60934448 | 17157.91796875 |
| 425.91204834 | 17159.54492188 |
...
```

We see the loss continue to decrease on the training set while the
loss on the test set is staying fairly constant.

## Early Stopping

Using the built-in XGBoost early stopping we can avoid overtraining:


```clojure
user> (def model (ml/train (:train-ds train-test-split)
                           (assoc model-options
                                  :watches train-test-split
                                  :eval-metric "mae"
                                  :round 100
                                  :early-stopping-round 4)))
09:34:35.063 [nREPL-session-d7c6e5a6-cc96-42d5-8435-99f1f16b6d1f] WARN tech.v3.libs.xgboost - Early stopping indicated but watches has undefined iteration order.
Early stopping will always use the 'last' of the watches as defined by the iteration
order of the watches map.  Consider using a java.util.LinkedHashMap for watches.
https://github.com/dmlc/xgboost/blob/master/jvm-packages/xgboost4j/src/main/java/ml/dml
c/xgboost4j/java/XGBoost.java#L208
#'user/model
```

Oops!  This is a implementation detail of xgboost.  We have to use a map that
retains insertion order in order to do early stopping or we have to have only
one watch.  XGBoost will use the *last entry* in the watches map to perform
it's early stopping checks.


```clojure
user> (import '[java.util LinkedHashMap])
java.util.LinkedHashMap
user> (def watches (doto (LinkedHashMap.)
                     (.put :train-ds (:train-ds train-test-split))
                     (.put :test-ds (:test-ds train-test-split))))
#'user/watches
user> (def model (ml/train (:train-ds train-test-split)
                           (assoc model-options
                                  :watches watches
                                  :eval-metric "mae"
                                  :round 100
                                  :early-stopping-round 4)))
#'user/model
user> (ds/tail (get-in model [:model-data :metrics]))
:metrics [5 2]:

| :train-ds | :test-ds |
|-----------|----------|
|       0.0 |      0.0 |
|       0.0 |      0.0 |
|       0.0 |      0.0 |
|       0.0 |      0.0 |
|       0.0 |      0.0 |
...
```
So here we asked XGBoost to stop if the error on the evaluation dataset (the 'last'
dataset in the watches map) rises 4 rounds consecutively.  As a result, we see that
the end of the metrics dataset is all zeros.  What round did we actually get to?

```clojure
user> (def final-metrics (-> (get-in model [:model-data :metrics])
                             (assoc :round (range 100))
                             (ds/filter-column :test-ds #(not= 0.0 (double %)))))

#'user/final-metrics
user> (ds/tail final-metrics)
:metrics [5 3]:

|     :train-ds |       :test-ds | :round |
|---------------|----------------|--------|
| 1799.49560547 | 17184.22460938 |     50 |
| 1751.87585449 | 17190.24023438 |     51 |
| 1715.23388672 | 17189.55859375 |     52 |
| 1690.89416504 | 17189.88085938 |     53 |
| 1653.31481934 | 17184.24023438 |     54 |
```

## Caveat - Cleaning the Options Map


The options used to train the model are stored verbatim in the model map.  This means
that if you use watches and then save the model you will probably more data in your
nippy than you had planned on and it might just not work at all:

```clojure
user> (require '[taoensso.nippy :as nippy])
nil
user> (count (nippy/freeze model))
Execution error (ExceptionInfo) at taoensso.nippy/throw-unfreezable (nippy.clj:982).
Unfreezable type: class java.util.LinkedHashMap
```

So we just clean up the options map (and the metrics) before saving:

```clojure
user> (-> (update model :model-data dissoc :metrics)
          (update :options dissoc :watches)
          (nippy/freeze)
          (count))
248398
```


## Conclusion


We grabbed a dataset, made it trainable (no missing, no strings, everything is
a numeric datatype).  We then trained an initial model and checked out which columns
XGBoost found useful.  We furthermore built out metrics using xgboost's watches
feature and observed error rates as XGBoost continued to train.  We then ask xgboost
to stop training if the error on a validation dataset (the last of the watches map
by iteration) increased 4 rounds consecutively.


For this dataset specifically there are quite a lot of dataset-specific operations
you can do to improve the results.  The examples above are intended to
show the range of training options that XGBoost provides out of the box.  We hope you
enjoy using this excellent software library and that these relatively simple techniques
can allow you to get to great results quickly.
