(ns tech.v3.libs.xgboost-test
  (:require [clojure.test :refer [deftest is]]
            [tech.v3.datatype.functional :as dfn]
            [tech.v3.ml :as ml]
            [tech.v3.ml.loss :as loss]
            [tech.v3.ml.verify :as verify]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.libs.xgboost]))


(deftest basic
  (verify/basic-regression {:model-type :xgboost/regression}))


(deftest basic-early-stopping
  (verify/basic-regression {:model-type :xgboost/regression
                            :early-stopping-round 5
                            :round 50}))

(deftest watches
  (let [{train-dataset :train-ds
         test-dataset :test-ds} (ds-mod/train-test-split @verify/regression-iris*)
        options {:model-type :xgboost/regression
                 :watches {:test-ds test-dataset}
                 :round 25
                 :eval-metric "mae"}
        model (ml/train train-dataset options)
        watch-data (get-in model [:model-data :metrics])
        predictions (ml/predict test-dataset model)
        mse (loss/mse (predictions verify/target-colname)
                      (test-dataset verify/target-colname))]
    (is (= 25 (ds/row-count watch-data)))
    (is (not= 0 (dfn/reduce-+ (watch-data :test-ds))))
    (is (< mse (double 0.2)))))


(deftest k-fold-regression
  (verify/k-fold-regression {:model-type :xgboost/regression}))


(deftest regression-gridsearch
  (verify/auto-gridsearch-regression {:model-type :xgboost/regression}))


(deftest classification
  (verify/basic-classification {:model-type :xgboost/classification}))


(deftest classification-gridsearch
  (verify/auto-gridsearch-classification {:model-type :xgboost/classification}))
