(ns tech.v3.ml.metamorph-test
  (:require [tech.v3.ml.metamorph :as sut]
            [clojure.test :refer [deftest is]]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset :as ds]
            [tech.v3.ml.loss :as loss]
            [tech.v3.libs.smile.classification]))


(deftest test-model
  (let [
        src-ds (ds/->dataset "test/data/iris.csv")
        ds (->  src-ds
                (ds/categorical->number cf/categorical)
                (ds-mod/set-inference-target "species")
                (ds/shuffle {:seed 1234}))
        feature-ds (cf/feature ds)
        split-data (ds-mod/train-test-split ds {:randomize-dataset? false})
        train-ds (:train-ds split-data)
        test-ds  (:test-ds split-data)

        pipeline (fn  [ctx]
                   ((sut/model {:model-type :smile.classification/random-forest})
                    ctx))


        fitted
        (pipeline
         {:metamorph/id "1"
          :metamorph/mode :fit
          :metamorph/data train-ds})


        prediction
        (pipeline (merge fitted
                         {:metamorph/mode :transform
                          :metamorph/data test-ds}))

        predicted-specis (ds-mod/column-values->categorical (:metamorph/data prediction)
                                                            "species"
                                                            )]

    (is (= ["setosa" "setosa" "virginica"]
           (take 3 predicted-specis)))

    ))
