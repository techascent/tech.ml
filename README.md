# tech.ml

[![Clojars Project](https://img.shields.io/clojars/v/techascent/tech.ml.svg)](https://clojars.org/techascent/tech.ml)

Library to encapsulate a few core concepts of techascent system.

## Core Concepts


### Dataset Pipeline Processing

Dataset ETL is a repeatable processing that stores data so that doing inference later is automatic.

1.  Build your ETL pipeline.
2.  Apply to training dataset.  Result is a new pipeline with things that min,max per column stored or even trained models.
3.  Train, gridsearch, get a model.
4.  Use ETL pipeline returned from (2) with no modification to apply to new inference samples.
5.  Infer.


Checkout the [unit tests](test/tech/libs/tablesaw_test.clj) and [example pipeline](example/src/tech/ml/example/svm_datasets.clj).


### ML Is Functional


Train is a function that takes a map of options and a sequence of data and returns a new map.
Nothing special about it aside from it figures out the subsystem from one of the keys in the
map of options.

The returned map contains a uuid ID so you can record your model ID somewhere and find it later.

(in the example project, but using code from the [classification verification](src/tech/verify/ml/classification.clj)


### Example


```clojure
user>
:tech.resource.gc Reference thread starting (require '[tech.verify.ml.classification :as classify-verify])
nil
user> (require '[tech.libs.xgboost])
nil
user> (require '[tech.ml :as ml])
nil
user> (require '[tech.ml.loss :as loss])
nil
user> (require '[tech.ml.dataset.etl :as etl])
nil
user> (require '[tech.ml.dataset :as dataset])
nil
user> (first (classify-verify/fruit-dataset))
{:color-score 0.55,
 :fruit-label 1.0,
 :fruit-name :apple,
 :fruit-subtype :granny-smith,
 :height 7.3,
 :mass 192.0,
 :width 8.4}


[[remove [:fruit-subtype :fruit-label]]
 [string->number string?]
 [range-scaler (not categorical?)]]
user> (def pipeline-result (etl/apply-pipeline (classify-verify/fruit-dataset)
                                               classify-verify/fruit-pipeline
                                               {:target :fruit-name}))
#'user/pipeline-result
user> (keys pipeline-result)
(:dataset :options :pipeline)
user> (:options pipeline-result)
{:dataset-column-metadata {:post-pipeline [{:categorical? true,
                                            :datatype :float64,
                                            :name :fruit-name,
                                            :size 59,
                                            :target? true}
                                           {:datatype :float64, :name :mass, :size 59}
                                           {:datatype :float64, :name :width, :size 59}
                                           {:datatype :float64, :name :height, :size 59}
                                           {:datatype :float64,
                                            :name :color-score,
                                            :size 59}],
                           :pre-pipeline [{:datatype :float32,
                                           :name :fruit-label,
                                           :size 59}
                                          {:categorical? true,
                                           :datatype :string,
                                           :name :fruit-name,
                                           :size 59}
                                          {:categorical? true,
                                           :datatype :string,
                                           :name :fruit-subtype,
                                           :size 59}
                                          {:datatype :float32, :name :mass, :size 59}
                                          {:datatype :float32, :name :width, :size 59}
                                          {:datatype :float32, :name :height, :size 59}
                                          {:datatype :float32,
                                           :name :color-score,
                                           :size 59}]},
 :feature-columns [:color-score :height :mass :width],
 :label-columns [:fruit-name],
 :label-map {:fruit-name {"apple" 0, "lemon" 2, "mandarin" 3, "orange" 1}},
 :target :fruit-name}
user> (:pipeline pipeline-result)
[{:context {}, :operation [remove [:fruit-subtype :fruit-label]]}
 {:context {:label-map {:fruit-name {"apple" 0, "lemon" 2, "mandarin" 3, "orange" 1}}},
  :operation [string->number (:fruit-name)]}
 {:context {:color-score {:max 0.9300000071525574, :min 0.550000011920929},
            :height {:max 10.5, :min 4.0},
            :mass {:max 362.0, :min 76.0},
            :width {:max 9.600000381469727, :min 5.800000190734863}},
  :operation [range-scaler #{:color-score :height :mass :width}]}]
user> (def model (ml/train (assoc (:options pipeline-result) :model-type :xgboost/classification)
                           (:dataset pipeline-result)))
#'user/model
user> (type model)
#<Class@ffaa6af clojure.lang.PersistentArrayMap>
user> (keys model)
(:model :options :id)


user> (def infer-pipeline (etl/apply-pipeline (classify-verify/fruit-dataset) (:pipeline pipeline-result) {:inference? true}))
[remove [:fruit-subtype :fruit-label]]
[string->number (:fruit-name)]
[range-scaler #{:mass :width :color-score :height}]
#'user/infer-pipeline
user> (ml/predict model (:dataset infer-pipeline))
({"apple" 0.98377246, "lemon" 0.0032576045, "mandarin" 0.003170099, "orange" 0.009799847}
 {"apple" 0.9763731, "lemon" 0.0032331028, "mandarin" 0.004000053, "orange" 0.016393797}
 {"apple" 0.97751075, "lemon" 0.0032699052, "mandarin" 0.003186134, "orange" 0.016033292}
 {"apple" 0.011603652, "lemon" 0.015576145, "mandarin" 0.93900126, "orange" 0.033818968}
 {"apple" 0.011314781, "lemon" 0.018377881, "mandarin" 0.9156251, "orange" 0.05468225}
 {"apple" 0.011117198, "lemon" 0.02829335, "mandarin" 0.899636, "orange" 0.06095348}
 {"apple" 0.018726224, "lemon" 0.018833136, "mandarin" 0.93830687, "orange" 0.024133723}
 {"apple" 0.018726224, "lemon" 0.018833136, "mandarin" 0.93830687, "orange" 0.024133723}
...
```

### Gridsearching

Gridsearching is often the best way to explore a dataset because you don't need to figure out
exactly how details of the dataset map to hyperparameters of the models.  Most models in the
tech.ml system allow gridsearching (xgboost certainly does):

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


user> (def gridsearch-results (ml/gridsearch (merge (:options pipeline-result)
                                                    {:k-fold 3}
                                                    (ml/auto-gridsearch-options {:model-type :xgboost/classification}))
                                             loss/classification-loss
                                             (:dataset pipeline-result)))
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
 (:model
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
 (:model
  :options
  :id
  :train-time
  :predict-time
  :loss
  :average-loss
  :total-train-time
  :total-predict-time))
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


[Here](example/src/tech/ml/example/classify.clj)  is an example doing just that.


## License

Copyright © 2018 Tech Ascent, LLC

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
