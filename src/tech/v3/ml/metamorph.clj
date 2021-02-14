(ns tech.v3.ml.metamorph
  (:require [tech.v3.ml :as ml]))

(defn model [options]
  (fn [{:metamorph/keys [id data mode] :as ctx}]
    (case mode
      :fit (assoc ctx id (ml/train data  options))
      :transform  (assoc ctx :metamorph/data (ml/predict data (get ctx id))))))
