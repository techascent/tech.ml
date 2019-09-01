(ns tech.libs.xgboost-test
  (:require [clojure.test :refer :all]
            [tech.verify.ml.regression :as verify-reg]
            [tech.verify.ml.classification :as verify-cls]
            [tech.v2.datatype :as dtype]
            [tech.v2.datatype.functional :as dfn]
            [tech.ml :as ml]
            [tech.ml.loss :as loss]
            [tech.ml.dataset :as dataset]
            [tech.libs.xgboost]))


(deftest basic
  (verify-reg/basic-regression {:model-type :xgboost/regression}))


(deftest basic-early-stopping
  (verify-reg/basic-regression {:model-type :xgboost/regression
                                :early-stopping-round 5
                                :round 50}))

(deftest watches
  (let [{train-dataset :train-ds
         test-dataset :test-ds} (verify-reg/datasets)
        options {:model-type :xgboost/regression
                 :watches {:test-ds test-dataset}
                 :round 25
                 :eval-metric "mae"}
        model (ml/train options train-dataset)
        watch-data (get-in model [:model :metrics :test-ds])
        test-output (ml/predict model test-dataset)
        mse (loss/mse test-output (dataset/labels test-dataset))]
    (is (= 25 (dtype/ecount watch-data)))
    (is (not= 0 (dfn/reduce-+ watch-data)))
    (is (not (nil? watch-data)))
    (is (< mse (double 0.2)))))


(deftest k-fold-regression
  (verify-reg/k-fold-regression {:model-type :xgboost/regression}))


(deftest regression-gridsearch
  (verify-reg/auto-gridsearch-simple {:model-type :xgboost/regression}))


(deftest classification
  (verify-cls/classify-fruit {:model-type :xgboost/classification}))


(deftest classification-gridsearch
  (verify-cls/auto-gridsearch-fruit {:model-type :xgboost/classification}))
