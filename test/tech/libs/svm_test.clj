(ns tech.svm-test
  (:require [clojure.test :refer :all]
            [tech.libs.svm :as svm]
            [tech.verify.ml.regression :as verify-regression]
            [tech.verify.ml.classification :as verify-classification]))


(deftest basic-regression
  (verify-regression/basic-regression {:model-type :libsvm/regression
                                       :accuracy 0.5}))

(deftest regression-gridsearch
  (verify-regression/auto-gridsearch-simple {:model-type :libsvm/regression}))


(deftest basic-classification
  (verify-classification/classify-fruit {:model-type :libsvm/classification}))


(deftest basic-classification-gridsearch
  (verify-classification/auto-gridsearch-fruit {:model-type :libsvm/classification}))
