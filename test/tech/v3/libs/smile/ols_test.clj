(ns tech.v3.libs.smile.ols-test
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.libs.smile.regression]
            [tech.v3.libs.smile.nlp :as nlp]
            [tech.v3.ml :as ml]
            [tech.v3.ml.gridsearch :as ml-gs]
            [clojure.test :refer (deftest is)]))

(def interest-rate  [2.75 2.5 2.5 2.5 2.5 2.5 2.5 2.25 2.25 2.25 2 2 2 1.75 1.75 1.75 1.75 1.75 1.75 1.75 1.75 1.75 1.75 1.75])
(def unemployment-rate [5.3 5.3 5.3 5.3 5.4 5.6 5.5 5.5 5.5 5.6 5.7 5.9 6 5.9 5.8 6.1 6.2 6.1 6.1 6.1 5.9 6.2 6.2 6.1])
(def stock-index-price [1464 1394 1357 1293 1256 1254 1234 1195 1159 1167 1130 1075 1047 965 943 958 971 949 884 866 876 822 704 719])

(defn absolute-difference ^double [^double x ^double y]
  (Math/abs (double (- x y))))

(defn close? [tolerance x y]
  (< (absolute-difference x y) tolerance))

(deftest explain
  (let [ds (-> (ds/->dataset {:interest-rate interest-rate
                              :unemployment-rate unemployment-rate
                              :stoc-index-price stock-index-price})
               (ds-mod/set-inference-target :stoc-index-price)
               )
        ols
        (ml/train ds {:model-type :smile.regression/ordinary-least-square})

        ols-model
        (ml/thaw-model ols
                       (ml/options->model-def (:options ols)))
        weights (ml/explain ols)]

    (is (close? 0.1 1798.4  (:bias weights)))
    (is (close? 0.1 345.5  (-> weights :coefficients first second)))
    (is (close? 0.1 -250.1  (-> weights :coefficients second second)))))
