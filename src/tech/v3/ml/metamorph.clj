(ns tech.v3.ml.metamorph
  (:require [tech.v3.ml :as ml]))


(defn model [options]
  (fn [ctx]
    (let [id (:metamorph/id ctx)
          ds (:metamorph/data ctx)
          mode (:metamorph/mode ctx)]
      (case mode
        :fit (assoc ctx id (ml/train ds  options))
        :transform  (assoc ctx :metamorph/data (ml/predict ds (get ctx id)))))))
