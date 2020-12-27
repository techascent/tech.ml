(ns tech.v3.libs.smile-ml-test
  (:require [tech.v3.ml.verify :as verify]
            [tech.v3.ml :as ml]
            [tech.v3.libs.smile.regression]
            [tech.v3.libs.smile.classification]
            [tech.v3.dataset.utils :as ds-utils]
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




;; take  long

;; (deftest smile-regression-autogridsearch-test
;;   (doseq [regression-model smile-regression-models]
;;     (verify/auto-gridsearch-regression {:model-type regression-model 0.5 })))


;; (deftest smile-classification-autogridsearch-test
;;   (doseq [classify-model smile-classification-models]
;;     (verify/auto-gridsearch-classification {:model-type classify-model} 0.5)))
