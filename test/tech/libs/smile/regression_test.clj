(ns tech.libs.smile.regression-test
  (:require [tech.verify.ml.regression :as verify-reg]
            [tech.libs.smile.regression]
            [clojure.test :refer :all]))

(deftest regression-basic
  (verify-reg/basic-regression {:model-type :smile.regression/lasso}))


(deftest regression-lasso-gridsearch
  (verify-reg/auto-gridsearch-simple {:model-type :smile.regression/lasso}))


(deftest regression-basic-ols
  (verify-reg/basic-regression {:model-type :smile.regression/ordinary-least-squares}))


(deftest regression-basic-ols-gridsearch
  (verify-reg/auto-gridsearch-simple
   {:model-type :smile.regression/ordinary-least-squares}))


(deftest regression-basic-rls
  (verify-reg/basic-regression {:model-type :smile.regression/recursive-least-squares}))


(deftest regression-basic-rls-gridsearch
  (verify-reg/auto-gridsearch-simple
   {:model-type :smile.regression/recursive-least-squares}))


(deftest regression-basic-svr
  (verify-reg/basic-regression {:model-type :smile.regression/support-vector
                                :accuracy 0.5}))


(deftest regression-basic-svr-gridsearch
  (verify-reg/auto-gridsearch-simple
   {:model-type :smile.regression/support-vector
    :accuracy 0.5}))


(deftest regression-basic-gradient-tree-boost
  (verify-reg/basic-regression {:model-type :smile.regression/gradient-tree-boost
                                :accuracy 1.2}))


(deftest regression-basic-gaussian-process
  (verify-reg/basic-regression {:model-type :smile.regression/gaussian-process
                                :accuracy 0.3}))


(deftest regression-basic-random-forest
  (verify-reg/basic-regression {:model-type :smile.regression/random-forest
                                :accuracy 2}))


(deftest elastic-net
  (verify-reg/basic-regression {:model-type :smile.regression/elastic-net
                                :accuracy 2}))


(deftest elastic-net-gridsearch
  (verify-reg/auto-gridsearch-simple {:model-type :smile.regression/elastic-net}))


(deftest ridge
  (verify-reg/basic-regression {:model-type :smile.regression/ridge}))


(deftest ridge-gridsearch
  (verify-reg/auto-gridsearch-simple {:model-type :smile.regression/ridge}))


(comment
  ;;Produces #NAN quite a bit, so *bad*!!
  (deftest regression-basic-nn
    (verify-reg/basic-regression :smile.regression :model-type :neural-network)))
