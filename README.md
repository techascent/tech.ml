# tech.ml

[![Clojars Project](https://img.shields.io/clojars/v/techascent/tech.ml.svg)](https://clojars.org/techascent/tech.ml)

Library to encapsulate a few core concepts of techascent system.

## Updates
* Smile now supports elasticnet in main distribution.  Use `{:model-type :smile.regression/elasticnet}` in your options.
* xgboost per-round error metrics are now [supported](https://github.com/techascent/tech.ml/blob/master/test/tech/libs/xgboost_test.clj#L22).



## Core Concepts


### Dataset Pipeline Processing

Dataset ETL is repeatable processing that stores data so that doing inference later is
automatic.

1.  Build your ETL pipeline.
2.  Apply to training dataset.  Result is a new pipeline with things that min,max per
    column stored or even trained models.
3.  Train, gridsearch, get a model.
4.  Build an inference pipeline using pipeline from step 1 with some augmentations.
5.  Infer.


Checkout the [example pipeline](example/src/tech/ml/example/svm_datasets.clj) and [shotgun classification approach](example/src/tech/ml/example/classify.clj).


### ML Is Functional


Train is a function that takes a map of options and a sequence of data and returns a new
map.  Nothing special about it aside from it figures out the subsystem from one of the
keys in the map of options.

The returned map contains a uuid ID so you can record your model ID somewhere and find
it later.

### Example


```clojure
user> (require '[tech.verify.ml.classification :as classify-verify])

:tech.resource.gc Reference thread starting
nil
user> (require '[tech.libs.xgboost])
nil
user> (require '[tech.ml :as ml])
nil
user> (require '[tech.ml.loss :as loss])
nil
user> (require '[tech.ml.dataset.pipeline :as dsp])
nil
user> (require '[tech.ml.dataset.pipeline.pipeline-operators
                 :refer [without-recording
                         pipeline-train-context
                         pipeline-inference-context]])
nil
user> (require '[tech.ml.dataset :as ds])
nil
user> (first (classify-verify/mapseq-dataset))
{:color-score 0.55,
 :fruit-label 1.0,
 :fruit-name :apple,
 :fruit-subtype :granny-smith,
 :height 7.3,
 :mass 192.0,
 :width 8.4}


user> (def fruits (ds/->dataset (classify-verify/mapseq-dataset)))
#'user/fruits

user> (require '[tech.v2.datatype :as dtype])
nil
user> (dtype/shape fruits)
[7 59]
user> (println (ds/select fruits :all (range 10)))
_unnamed [10 7]:

| :fruit-label | :fruit-name | :fruit-subtype |   :mass | :width | :height | :color-score |
|--------------+-------------+----------------+---------+--------+---------+--------------|
|        1.000 |       apple |   granny-smith | 192.000 |  8.400 |   7.300 |        0.550 |
|        1.000 |       apple |   granny-smith | 180.000 |  8.000 |   6.800 |        0.590 |
|        1.000 |       apple |   granny-smith | 176.000 |  7.400 |   7.200 |        0.600 |
|        2.000 |    mandarin |       mandarin |  86.000 |  6.200 |   4.700 |        0.800 |
|        2.000 |    mandarin |       mandarin |  84.000 |  6.000 |   4.600 |        0.790 |
|        2.000 |    mandarin |       mandarin |  80.000 |  5.800 |   4.300 |        0.770 |
|        2.000 |    mandarin |       mandarin |  80.000 |  5.900 |   4.300 |        0.810 |
|        2.000 |    mandarin |       mandarin |  76.000 |  5.800 |   4.000 |        0.810 |
|        1.000 |       apple |       braeburn | 178.000 |  7.100 |   7.800 |        0.920 |
|        1.000 |       apple |       braeburn | 172.000 |  7.400 |   7.000 |        0.890 |

nil

user> (require '[tech.ml.dataset.pipeline.column-filters :as cf])
nil
user> (defn fruit-pipeline
  [dataset]
  (-> dataset
      (ds/remove-columns [:fruit-subtype :fruit-label])
      (dsp/string->number)
      (dsp/range-scale #(cf/not cf/categorical?))
      (ds/set-inference-target :fruit-name)))
#'user/fruit-pipeline
user> (def processed-ds (fruit-pipeline fruits))
#'user/processed-ds
user> (println (ds/select processed-ds :all (range 10)))
_unnamed [10 5]:

| :fruit-name |  :mass | :width | :height | :color-score |
|-------------+--------+--------+---------+--------------|
|       0.000 | -0.189 |  0.368 |   0.015 |       -1.000 |
|       0.000 | -0.273 |  0.158 |  -0.138 |       -0.789 |
|       0.000 | -0.301 | -0.158 |  -0.015 |       -0.737 |
|       3.000 | -0.930 | -0.789 |  -0.785 |        0.316 |
|       3.000 | -0.944 | -0.895 |  -0.815 |        0.263 |
|       3.000 | -0.972 | -1.000 |  -0.908 |        0.158 |
|       3.000 | -0.972 | -0.947 |  -0.908 |        0.368 |
|       3.000 | -1.000 | -1.000 |  -1.000 |        0.368 |
|       0.000 | -0.287 | -0.316 |   0.169 |        0.947 |
|       0.000 | -0.329 | -0.158 |  -0.077 |        0.789 |

nil


user> (def model (ml/train {:model-type :xgboost/classification}
                           processed-ds))
#'user/model
user> (type model)
#<Class@ffaa6af clojure.lang.PersistentArrayMap>
user> (keys model)
(:model :options :id)
user> (:options model)
{:model-type :xgboost/classification,
 :dataset-shape [5 59],
 :feature-columns [:mass :width :height :color-score],
 :label-columns [:fruit-name],
 :label-map {:fruit-name {"apple" 0, "orange" 1, "lemon" 2, "mandarin" 3}},
 :column-map
 {:mass {:name :mass, :size 59, :datatype :float64, :column-type :feature},
  :width {:name :width, :size 59, :datatype :float64, :column-type :feature},
  :height {:name :height, :size 59, :datatype :float64, :column-type :feature},
  :color-score
  {:name :color-score, :size 59, :datatype :float64, :column-type :feature},
  :fruit-name
  {:name :fruit-name,
   :categorical? true,
   :size 59,
   :datatype :float64,
   :label-map {"apple" 0, "orange" 1, "lemon" 2, "mandarin" 3},
   :column-type :inference}}}


;; Note that the system takes care of the inverse label map from fruit-name back to the
;; categorical value.  The underlying columnstore table system only supports strings so
;; our results are in strings and not keywords.
;; The key takeaway though is that the label mapping is stored with the model so you
;; cannot possibly get into a situation where your labels do not match your model.

user> (take 10 (ml/predict model processed-ds))
({"apple" 0.98378086,
  "orange" 0.010113608,
  "lemon" 0.0028871458,
  "mandarin" 0.0032183384}
 {"apple" 0.975659, "orange" 0.016668763, "lemon" 0.0028633103, "mandarin" 0.00480895}
 {"apple" 0.97181576,
  "orange" 0.019529564,
  "lemon" 0.0037236277,
  "mandarin" 0.004931037}
 {"apple" 0.01430875, "orange" 0.035686996, "lemon" 0.022270069, "mandarin" 0.9277342}
 {"apple" 0.014057106, "orange" 0.057201006, "lemon" 0.01732342, "mandarin" 0.91141844}
 {"apple" 0.012475298, "orange" 0.06429278, "lemon" 0.017220644, "mandarin" 0.9060113}
 {"apple" 0.018275188, "orange" 0.02577525, "lemon" 0.017830912, "mandarin" 0.93811864}
 {"apple" 0.018275188, "orange" 0.02577525, "lemon" 0.017830912, "mandarin" 0.93811864}
 {"apple" 0.9520015, "orange" 0.02614804, "lemon" 0.01359778, "mandarin" 0.008252703}
 {"apple" 0.98126006,
  "orange" 0.0124242315,
  "lemon" 0.0016809247,
  "mandarin" 0.0046347086})



;; Now we actually measure what we got against what we want.  Because we trained
;; on the dataset that we are measuring against (we didn't split it up in any way)
;; we get a perfect score.  Note that accuracy is the opposite of loss; accuracy
;; goes up as you get better while loss goes down.

user> (def test-output (ml/predict model processed-ds))
#'user/test-output
user> (def labels (ds/labels processed-ds))
#'user/labels
user> (take 5 labels)
("apple" "apple" "apple" "mandarin" "mandarin")
user> (take 5 test-output)
({"apple" 0.98378086,
  "orange" 0.010113608,
  "lemon" 0.0028871458,
  "mandarin" 0.0032183384}
 {"apple" 0.975659, "orange" 0.016668763, "lemon" 0.0028633103, "mandarin" 0.00480895}
 {"apple" 0.97181576,
  "orange" 0.019529564,
  "lemon" 0.0037236277,
  "mandarin" 0.004931037}
 {"apple" 0.01430875, "orange" 0.035686996, "lemon" 0.022270069, "mandarin" 0.9277342}
 {"apple" 0.014057106, "orange" 0.057201006, "lemon" 0.01732342, "mandarin" 0.91141844})
user> (require '[tech.ml.loss :as loss])
nil
user> (loss/classification-accuracy test-output labels)
1.0
user> (loss/classification-loss test-output labels)
0.0

;; Wash, rinse repeat.  Do your feature engineering until you get the outcomes
;; that you want.

;; The immediate next problem is how do you take what you have and put it into production.
;; You have a model, or rather a process for generating an acceptible model.  Now we need
;; to codify this process such that it produces both a model and some pipeline context.


;; We redefine our pipeline such that the processing that should only
;; occur in training time does in fact only occur during training.


user> (defn fruit-pipeline
  [dataset training?]
  (-> dataset
      (ds/remove-columns [:fruit-subtype :fruit-label])
      (dsp/range-scale #(cf/not cf/categorical?))
      (dsp/pwhen
       training?
       #(without-recording
         (-> %
             (dsp/string->number :fruit-name)
             (ds/set-inference-target :fruit-name))))))

#'user/fruit-pipeline


;; We then 'train' our pipeline on the training data producing both
;; a training dataset to train a model and some context that will
;; be used during inference.

(def dataset-train-data (pipeline-train-context
                           (fruit-pipeline fruits true)))
#'user/dataset-train-data
user> (keys dataset-train-data)
(:context :dataset)
user> (:context dataset-train-data)
{:pipeline-environment {},
 :operator-context
 [{:column-name-seq (:mass :width :height :color-score),
   :context
   {:mass {:min 76.0, :max 362.0},
    :width {:min 5.800000190734863, :max 9.600000381469727},
    :height {:min 4.0, :max 10.5},
    :color-score {:min 0.550000011920929, :max 0.9300000071525574}}}]}
user> (println (ds/select (:dataset dataset-train-data) :all (range 10)))
_unnamed [10 5]:

| :fruit-name |  :mass | :width | :height | :color-score |
|-------------+--------+--------+---------+--------------|
|       0.000 | -0.189 |  0.368 |   0.015 |       -1.000 |
|       0.000 | -0.273 |  0.158 |  -0.138 |       -0.789 |
|       0.000 | -0.301 | -0.158 |  -0.015 |       -0.737 |
|       3.000 | -0.930 | -0.789 |  -0.785 |        0.316 |
|       3.000 | -0.944 | -0.895 |  -0.815 |        0.263 |
|       3.000 | -0.972 | -1.000 |  -0.908 |        0.158 |
|       3.000 | -0.972 | -0.947 |  -0.908 |        0.368 |
|       3.000 | -1.000 | -1.000 |  -1.000 |        0.368 |
|       0.000 | -0.287 | -0.316 |   0.169 |        0.947 |
|       0.000 | -0.329 | -0.158 |  -0.077 |        0.789 |


;;Now imagine we are in production.  Our dataset will not have the
;;answers in it, so it will look more like:


user> (def inference-src-ds (ds/remove-columns fruits [:fruit-name :fruit-subtype :fruit-label]))
#'user/inference-src-ds
user> (println (ds/select inference-src-ds :all (range 10)))
_unnamed [10 4]:

|   :mass | :width | :height | :color-score |
|---------+--------+---------+--------------|
| 192.000 |  8.400 |   7.300 |        0.550 |
| 180.000 |  8.000 |   6.800 |        0.590 |
| 176.000 |  7.400 |   7.200 |        0.600 |
|  86.000 |  6.200 |   4.700 |        0.800 |
|  84.000 |  6.000 |   4.600 |        0.790 |
|  80.000 |  5.800 |   4.300 |        0.770 |
|  80.000 |  5.900 |   4.300 |        0.810 |
|  76.000 |  5.800 |   4.000 |        0.810 |
| 178.000 |  7.100 |   7.800 |        0.920 |
| 172.000 |  7.400 |   7.000 |        0.890 |

nil

user> (def inference-data (pipeline-inference-context
                           (:context dataset-train-data)
                           (fruit-pipeline inference-src-ds false)))
#'user/inference-data
user>
user> (keys inference-data)
(:dataset)
user> (println (ds/select (:dataset inference-data) :all (range 10)))
_unnamed [10 4]:

|  :mass | :width | :height | :color-score |
|--------+--------+---------+--------------|
| -0.189 |  0.368 |   0.015 |       -1.000 |
| -0.273 |  0.158 |  -0.138 |       -0.789 |
| -0.301 | -0.158 |  -0.015 |       -0.737 |
| -0.930 | -0.789 |  -0.785 |        0.316 |
| -0.944 | -0.895 |  -0.815 |        0.263 |
| -0.972 | -1.000 |  -0.908 |        0.158 |
| -0.972 | -0.947 |  -0.908 |        0.368 |
| -1.000 | -1.000 |  -1.000 |        0.368 |
| -0.287 | -0.316 |   0.169 |        0.947 |
| -0.329 | -0.158 |  -0.077 |        0.789 |

nil

user> (take 10 (ml/predict model (:dataset inference-data)))
({"apple" 0.98378086,
  "orange" 0.010113608,
  "lemon" 0.0028871458,
  "mandarin" 0.0032183384}
 {"apple" 0.975659, "orange" 0.016668763, "lemon" 0.0028633103, "mandarin" 0.00480895}
 {"apple" 0.97181576,
  "orange" 0.019529564,
  "lemon" 0.0037236277,
  "mandarin" 0.004931037}
 {"apple" 0.01430875, "orange" 0.035686996, "lemon" 0.022270069, "mandarin" 0.9277342}
 {"apple" 0.014057106, "orange" 0.057201006, "lemon" 0.01732342, "mandarin" 0.91141844}
 {"apple" 0.012475298, "orange" 0.06429278, "lemon" 0.017220644, "mandarin" 0.9060113}
 {"apple" 0.018275188, "orange" 0.02577525, "lemon" 0.017830912, "mandarin" 0.93811864}
 {"apple" 0.018275188, "orange" 0.02577525, "lemon" 0.017830912, "mandarin" 0.93811864}
 {"apple" 0.9520015, "orange" 0.02614804, "lemon" 0.01359778, "mandarin" 0.008252703}
 {"apple" 0.98126006,
  "orange" 0.0124242315,
  "lemon" 0.0016809247,
  "mandarin" 0.0046347086})
...
```

### Gridsearching

Gridsearching is often the best way to explore a dataset because you don't need to
figure out exactly how details of the dataset map to hyperparameters of the models.
Most models in the tech.ml system allow gridsearching (xgboost certainly does):

```clojure

(def options {:model-type :xgboost/classification})
#'user/options
user> (ml/auto-gridsearch-options options)
{:alpha #<Fn@52d920c tech.ml.gridsearch/make_gridsearch_fn[fn]>,
 :eta #<Fn@381ac7e6 tech.ml.gridsearch/make_gridsearch_fn[fn]>,
 :gamma #<Fn@7e1e1a1 tech.ml.gridsearch/make_gridsearch_fn[fn]>,
 :lambda #<Fn@68373970 tech.ml.gridsearch/make_gridsearch_fn[fn]>,
 :max-depth #<Fn@1636b88 clojure.core/comp[fn]>,
 :model-type :xgboost/classification,
 :scale-pos-weight #<Fn@6fa8b9b5 tech.ml.gridsearch/make_gridsearch_fn[fn]>,
 :subsample #<Fn@67cca108 tech.ml.gridsearch/make_gridsearch_fn[fn]>}
 ```

We then just do k-fold across a range of options:

```clojure


user> (def gridsearch-results (ml/gridsearch (merge {:k-fold 3}
                                                    (ml/auto-gridsearch-options
                                                     {:model-type :xgboost/classification}))
                                             loss/classification-loss
                                             processed-ds))
#'user/gridsearch-results

user> (count gridsearch-results)
5
user> (map :average-loss gridsearch-results)
(0.03508771929824561
 0.05175438596491229
 0.05263157894736843
 0.05263157894736843
 0.05263157894736843)
user> (map keys gridsearch-results)
((:model
  :options
  :id
  :train-time
  :predict-time
  :loss
  :average-loss
  :total-train-time
  :total-predict-time)
 (:model
  :options
  :id
  :train-time
  :predict-time
  :loss
  :average-loss
  :total-train-time
  :total-predict-time)
  ...)
```

Using the results of this, we get a sort list of the best models.
You can then use the options produced via gridsearching to re-train some number of these
models and then just take the best one or do an ensemble with ones that are uncorrelated
across some dimensions you care about.

We can also graph the relationship between various hyperparameters and the loss as well
as between various model types and hyperparameters and the training or prediction times.


### Concluding


We have generic support for xgboost and smile.  This gives you quite a few models and
they are all gridsearcheable as above.  We put this forward in an attempt to simplify
doing ML that we do and in an attempt to move the Clojure ML conversation forward
towards getting the best possible results for a dataset in the least amount of
(developer) time.


[Here](https://github.com/cnuernber/ames-house-prices/blob/master/src/clj_ml_wkg/ames_house_prices.clj)  is an example doing just that.


## License

Copyright © 2019 Tech Ascent, LLC

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
