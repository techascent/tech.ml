(ns tech.xgboost-test
  (:require [clojure.test :refer :all]
            [tech.verify.ml.regression :as verify-reg]
            [tech.verify.ml.classification :as verify-cls]
            [tech.xgboost]))


(deftest basic
  (verify-reg/basic-regression {:model-type :xgboost/regression}))


(deftest basic-early-stopping
  (verify-reg/basic-regression {:model-type :xgboost/regression
                                :early-stopping-round 5
                                :round 50}))


(deftest scaled-features
  (verify-reg/scaled-features {:model-type :xgboost/regression}))


(deftest k-fold-regression
  (verify-reg/k-fold-regression {:model-type :xgboost/regression}))


(deftest regression-gridsearch
  (verify-reg/auto-gridsearch-simple {:model-type :xgboost/regression}))


(deftest classification
  (verify-cls/classify-fruit {:model-type :xgboost/classification}))


(deftest classification-gridsearch
  (verify-cls/auto-gridsearch-fruit {:model-type :xgboost/classification}))
