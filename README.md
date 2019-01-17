# tech.ml-base

[![Clojars Project](https://img.shields.io/clojars/v/techascent/tech.ml-base.svg)](https://clojars.org/techascent/tech.ml-base)

Library to encapsulate a few core concepts of techascent system.

## Core Concepts


### Datasets Are Sequences Of Maps

We have found that for most problems, including computer vision, a sequence of maps is a great abstraction for a dataset:
```clojure
;; This is a dataset
(def test-ds [{:a 1 :b 2} {:a 3 :b 4}])
```

Before the dataset gets to the particular ML subsystem, we use the optimized system in
[tech.datatype](https://github.com/techascent/tech.datatype) to convert this style of dataset into
a sequence of maps with fewer keys and aggregate data:

```clojure
(def optimized-ds [{:tech.ml.dataset/features (double-array [1 2])} {:tech.ml.dataset/features (double-array [3 4])}])
```

Note that this subsystem is capable of dealing with native pointer based things like
[opencv images](http://techascent.com/blog/opencv-love.html).  So those can be entries in your dataset and everything
will work out fine.



### ML Is Functional


Train is a function that takes a map of options and a sequence of data and returns a new map.
Nothing special about it aside from it figures out the subsystem from one of the keys in the
map of options.

The returned map contains a uuid ID so you can record your model ID somewhere and find it later.

(in the example project, but using code from the [classification verification](src/tech/verify/ml/classification.clj)
```clojure
user> (require '[tech.verify.ml.classification :as classify-verify])

user> (require '[tech.xgboost])
nil
user> (require '[tech.ml-base :as ml])
nil
user> (require '[tech.ml.loss :as loss])

user> (first (classify-verify/fruit-dataset))

{:color-score 0.55,
 :fruit-label 1.0,
 :fruit-name :apple,
 :fruit-subtype :granny-smith,
 :height 7.3,
 :mass 192.0,
 :width 8.4}



user> (require '[tech.xgboost])
nil
user> (require '[tech.ml-base :as ml])
nil
user> (require '[tech.ml.loss :as loss])

nil
user> (require '[tech.ml.dataset :as dataset])
nil
user> (def split-ds  (->> (classify-verify/fruit-dataset)
                          (dataset/->train-test-split {})))
#'user/split-ds
user> (def train-ds (:train-ds split-ds))
#'user/train-ds

user> (ml/train  {:model-type :xgboost/classification}
                 [:color-score :height :mass :width]
                 :fruit-name
                 train-ds)

{:feature-keys [:color-score :height :mass :width],
 :id #uuid "ceb44288-8c02-413d-93af-c649dccb63c5",
 :label-keys :fruit-name,
 :model #<[B@693e6632>,
 :options {:container-fn #<Fn@7d8aa28a tech.datatype/make_array_of_type>,
           :datatype :float32,
           :label-map {:fruit-name {:apple 0, :lemon 2, :mandarin 3, :orange 1}},
           :model-type :xgboost/classification,
           :multiclass-label-base-index 0,
           :tech.ml.dataset/dataset-info {:tech.ml.dataset/feature-ecount 4,
                                          :tech.ml.dataset/key-ecount-map {:color-score 1,
                                                                           :fruit-name 1,
                                                                           :height 1,
                                                                           :mass 1,
                                                                           :width 1},
                                          :tech.ml.dataset/num-classes 4},
           :tech.ml.dataset/feature-keys [:color-score :height :mass :width],
           :tech.ml.dataset/label-keys [:fruit-name]}}
           
user> (def model *1)
#'user/model
```

Predict logically takes the output of train and a sequence of data and does a prediction.

```clojure

user> (ml/predict model (:test-ds split-ds))
({:apple 0.93956864, :lemon 0.0131554315, :mandarin 0.008915106, :orange 0.03836079}
 {:apple 0.13050404, :lemon 0.0260633, :mandarin 0.036519047, :orange 0.8069136}
 {:apple 0.20915678, :lemon 0.0561468, :mandarin 0.078457534, :orange 0.6562389}
 {:apple 0.01701929, :lemon 0.015879799, :mandarin 0.01436161, :orange 0.95273936}
 {:apple 0.005177637, :lemon 0.9728976, :mandarin 0.009348835, :orange 0.012575978}
 {:apple 0.04849192, :lemon 0.18590978, :mandarin 0.08649346, :orange 0.67910486}
 {:apple 0.93960667, :lemon 0.013155963, :mandarin 0.008875069, :orange 0.038362343}
 {:apple 0.15794337, :lemon 0.6966642, :mandarin 0.038001172, :orange 0.10739119}
 {:apple 0.85359246, :lemon 0.046843067, :mandarin 0.016741931, :orange 0.08282253}
 {:apple 0.9050958, :lemon 0.021029698, :mandarin 0.018952616, :orange 0.05492185}
 {:apple 0.35707343, :lemon 0.090002134, :mandarin 0.12557973, :orange 0.4273447}
 {:apple 0.0112614175, :lemon 0.94016343, :mandarin 0.008290613, :orange 0.04028459}
 {:apple 0.10490781, :lemon 0.4778567, :mandarin 0.15647744, :orange 0.260758}
 {:apple 0.07708386, :lemon 0.47674534, :mandarin 0.022594173, :orange 0.4235766}
 {:apple 0.006770351, :lemon 0.970588, :mandarin 0.0096611455, :orange 0.012980503}
 {:apple 0.9366283, :lemon 0.011120986, :mandarin 0.015428788, :orange 0.03682192}
 {:apple 0.93760806, :lemon 0.010915581, :mandarin 0.015294555, :orange 0.036181804}
 {:apple 0.010260959, :lemon 0.92626655, :mandarin 0.008225733, :orange 0.055246774})

 user> (def predictions *1)
#'user/predictions
user> (def labels (map :fruit-name (:test-ds split-ds)))
#'user/labels
user> (loss/classification-accuracy predictions labels)
0.7222222222222222
```

Note that because everything is specified completely in the output of train, you don't need to
know anything extra to predict.  The keys used to train, any dataset normalization procedures,
all of this is stored in the map returned by train.


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

user> (ml/gridsearch [gs-options]
                     [:color-score :height :mass :width]
                     :fruit-name
                     loss/classification-loss (classify-verify/fruit-dataset)
                     ;;Small k-fold because tiny dataset
                     :k-fold 3)
({:average-loss 0.050877192982456146,
  :k-fold 3,
  :options {:alpha 0.14358876879825508,
            :eta 0.796875,
            :gamma 0.0017707771425311252,
            :k-fold 3,
            :label-map {:fruit-name {:apple 0, :lemon 3, :mandarin 1, :orange 2}},
            :lambda 2.109375,
            :max-depth 40,
            :model-type :xgboost/classification,
            :scale-pos-weight 0.3671875,
            :subsample 0.47968750000000004,
            :tech.ml.dataset/dataset-info {:tech.ml.dataset/feature-ecount 4,
                                           :tech.ml.dataset/key-ecount-map {:color-score 1,
                                                                            :fruit-name 1,
                                                                            :height 1,
                                                                            :mass 1,
                                                                            :width 1},
                                           :tech.ml.dataset/num-classes 4},
            :tech.ml.dataset/feature-keys [:color-score :height :mass :width],
            :tech.ml.dataset/label-keys [:fruit-name]},
  :predict-time 4,
  :train-time 150}
 {:average-loss 0.05087719298245619,
  :k-fold 3,
  :options {:alpha 0.022360679774997897,
            :eta 0.5,
            :gamma 0.022360679774997897,
            :k-fold 3,
            :label-map {:fruit-name {:apple 0, :lemon 3, :mandarin 1, :orange 2}},
            :lambda 2.5,
            :max-depth 251,
            :model-type :xgboost/classification,
            :scale-pos-weight 1.05,
            :subsample 0.55,
            :tech.ml.dataset/dataset-info {:tech.ml.dataset/feature-ecount 4,
                                           :tech.ml.dataset/key-ecount-map {:color-score 1,
                                                                            :fruit-name 1,
                                                                            :height 1,
                                                                            :mass 1,
                                                                            :width 1},
                                           :tech.ml.dataset/num-classes 4},
            :tech.ml.dataset/feature-keys [:color-score :height :mass :width],
            :tech.ml.dataset/label-keys [:fruit-name]},
  :predict-time 4,
  :train-time 200}

  ...)
```

Using the results of this, we get a sort list of the best models *without* the model.
You can then use the options produced via gridsearching to train some number of these
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


[Here](example/src/tech/ml/classify.clj)  is an example doing just that.


## License

Copyright © 2018 Tech Ascent, LLC

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
