(ns tech.ml.loss
  (:require [clojure.core.matrix :as m]))


(defn mse
  "mean squared error"
  ^double [predictions labels]
  (assert (= (count predictions) (count labels)))
  (let [n (count predictions)]
    (double
     (/ (-> (m/sub predictions labels)
            (m/pow 2)
            (m/esum))
        n))))
