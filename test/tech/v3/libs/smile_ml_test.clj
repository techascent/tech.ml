(ns tech.v3.libs.smile-ml-test
  (:require [tech.v3.ml.verify :as verify]
            [tech.v3.ml :as ml]
            [tech.v3.libs.smile.regression]
            [tech.v3.libs.smile.classification]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.utils :as ds-utils]
            [tech.v3.datatype :as dtype]
            [tech.v3.dataset.column-filters :as cf]
            [clojure.test :refer [deftest is]]))

;;shut that shit up.
(ds-utils/set-slf4j-log-level :warn)


(def smile-regression-models
  (->> (ml/model-definition-names)
       (filter #(= "smile.regression" (namespace %)))))


(deftest smile-regression-test
  (doseq [reg-model smile-regression-models]
    (verify/basic-regression {:model-type reg-model})))


(def smile-classification-models
  (->> (ml/model-definition-names)
       (filter #(= "smile.classification" (namespace %)))))


(deftest smile-classification-test
  (doseq [classify-model smile-classification-models]
    (verify/basic-classification {:model-type classify-model})))




(deftest smile-regression-autogridsearch-test
    (doseq [regression-model smile-regression-models]
      (verify/auto-gridsearch-regression {:model-type regression-model} 0.5)))


(deftest smile-classification-autogridsearch-test
    (doseq [classify-model smile-classification-models]
      (verify/auto-gridsearch-classification {:model-type classify-model} 0.5)))


(deftest test-require-categorical-target
  (let [titanic (-> (ds/->dataset "test/data/titanic.csv")
                    (ds/drop-columns ["Name"])
                    (ds-mod/set-inference-target "Survived"))

        titanic-numbers (ds/categorical->number titanic cf/categorical)
        split-data (ds-mod/train-test-split titanic-numbers)
        train-ds (:train-ds split-data)
        test-ds (:test-ds split-data)
        ]
    (def test-ds test-ds)
     (is (thrown? Exception
                   (ml/train train-ds {:model-type :smile.classification/random-forest})
                   ))

    ))
