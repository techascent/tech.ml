(ns tech.libs.smile.classification-test
  (:require [tech.verify.ml.classification :as verify-cls]
            ;; [tech.libs.smile.classification]
            [tech.ml.utils :as utils]
            [tech.ml.dataset :as ds]
            [tech.ml.dataset.pipeline :as dsp]
            [tech.ml :as ml]
            [tech.libs.smile.protocols :as smile-proto]
            [clojure.test :refer :all])
  (:import [ch.qos.logback.classic Logger]
           [ch.qos.logback.classic Level]))


(utils/set-slf4j-log-level :warn)

(deftest ada-boost
  (verify-cls/classify-fruit {:model-type :smile.classification/ada-boost}))


;; (deftest ada-boost-gridsearch
;;   (verify-cls/auto-gridsearch-fruit {:model-type :smile.classification/ada-boost}))


;; (deftest fld
;;   (verify-cls/classify-fruit {:model-type :smile.classification/fld
;;                               :classification-accuracy 0.4}))


;; (deftest knn
;;   (verify-cls/classify-fruit {:model-type :smile.classification/knn
;;                               :classification-accuracy 0.5}))


;; (deftest knn-gridsearch
;;   (verify-cls/auto-gridsearch-fruit {:model-type :smile.classification/knn
;;                                      :classification-accuracy 0.5}))


;; This one is all over the board.  There are a few hyperparameters
;; that probably need to be tuned.
;; (deftest svm
;;   (verify-cls/classify-fruit {:model-type :smile.classification/svm
;;                               :classification-accuracy 0.01}))


;; (deftest svm-gridsearch
;;   (verify-cls/auto-gridsearch-fruit {:model-type :smile.classification/svm
;;                                      :classification-loss 0.30}))


;; (deftest svm-binary
;;   (testing "binary classifiers do not throw"
;;     (let [dataset (-> (ds/->dataset (concat (repeat 10 {:features 1 :label :a})
;;                                             (repeat 10 {:features -1 :label :b})))
;;                       (dsp/string->number)
;;                       (ds/set-inference-target :label))
;;           model (ml/train {:model-type :smile.classification/svm} dataset)]
;;       (is (not (nil? model))))))


;; (deftest lda
;;   (verify-cls/classify-fruit {:model-type :smile.classification/linear-discriminant-analysis
;;                               :classification-accuracy 0.4}))


;; (deftest lda-gridsearch
;;   (verify-cls/auto-gridsearch-fruit {:model-type :smile.classification/linear-discriminant-analysis
;;                                      :classification-loss 0.5}))


;;QDA won't work on the fruit example, not sure why at this point.
;; (deftest qda
;;   (verify-cls/classify-fruit
;;    :smile.classification {:model-type :quadratic-discriminant-analysis}))


;; (deftest qda-gridsearch
;;   (verify-cls/auto-gridsearch-fruit
;;    :smile.classification {:model-type :quadratic-discriminant-analysis}))


;; This one is also all over the place.  Given the right problem this is a great
;; model; just not appropriate for our dataset.x
;; (deftest rda
;;   (verify-cls/classify-fruit {:model-type :smile.classification/regularized-discriminant-analysis
;;                               :classification-accuracy 0.5}))


;; (deftest rda-gridsearch
;;   (verify-cls/auto-gridsearch-fruit {:model-type :smile.classification/regularized-discriminant-analysis}))


;; (deftest logistic-regression
;;   (verify-cls/classify-fruit {:model-type :smile.classification/logistic-regression
;;                               :classification-accuracy 0.5}))


;; (deftest logistic-regression-gridsearch
;;   (verify-cls/auto-gridsearch-fruit {:model-type :smile.classification/logistic-regression
;;                                      :classification-loss 0.30}))


;; ;;Default naive-bayes completely fails.
;; (deftest naive-bayes
;;   (verify-cls/classify-fruit {:model-type :smile.classification/naive-bayes
;;                               :classification-accuracy 0}))


;; (deftest naive-bayes-gridsearch
;;   (verify-cls/auto-gridsearch-fruit {:model-type :smile.classification/naive-bayes
;;                                      :classification-loss 0.6}))
