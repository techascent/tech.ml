(ns tech.libs.smile.regression-test
  (:require [tech.verify.ml.regression :as verify-reg]
            [tech.libs.smile.regression]
            [tech.ml.utils :as utils]
            [clojure.test :refer :all]))

(utils/set-slf4j-log-level :warn)


(deftest regression-basic
  (verify-reg/basic-regression {:model-type :smile.regression/lasso}))


(deftest regression-lasso-gridsearch
  (verify-reg/auto-gridsearch-simple {:model-type :smile.regression/lasso}))


(deftest ridge
  (verify-reg/basic-regression {:model-type :smile.regression/ridge}))


(deftest ridge-gridsearch
  (verify-reg/auto-gridsearch-simple {:model-type :smile.regression/ridge}))


(deftest elastic-net
  (verify-reg/basic-regression {:model-type :smile.regression/elastic-net
                                :accuracy 2}))


(deftest elastic-net-gridsearch
  (verify-reg/auto-gridsearch-simple {:model-type :smile.regression/elastic-net}))


(deftest regression-basic-gradient-tree-boost
  (verify-reg/basic-regression {:model-type :smile.regression/gradient-tree-boost
                                :accuracy 1.2}))


;; (deftest regression-basic-gaussian-process
;;   (verify-reg/basic-regression {:model-type :smile.regression/gaussian-process
;;                                 :accuracy 0.3}))


;; (deftest regression-basic-random-forest
;;   (verify-reg/basic-regression {:model-type :smile.regression/random-forest
;;                                 :accuracy 2}))


(comment
  ;;Produces #NAN quite a bit, so *bad*!!
  (deftest regression-basic-nn
    (verify-reg/basic-regression :smile.regression :model-type :neural-network)))
