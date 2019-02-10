(ns tech.ml.dataset.etl.math-ops
  (:require [tech.ml.dataset.etl.defaults :refer [etl-datatype]]
            [tech.ml.dataset.column :as ds-col]
            [tech.ml.dataset :as ds]))


(defonce ^:dynamic *etl-math-ops* (atom {}))


(defn register-math-op!
  [op-kwd op-type op-fn]
  (swap! *etl-math-ops* assoc op-kwd {:type op-type
                                      :operand op-fn})
  (keys @*etl-math-ops*))



(defn get-operand
  "Return a map of (at least)
  {:type op-type
   :operand op-fn
  }"
  [op-kwd]
  (if-let [retval (get @*etl-math-ops* op-kwd)]
    retval
    (throw (ex-info (format "Failed to find math operand: %s" op-kwd)
                    {:operand op-kwd}))))


(register-math-op!
 :col :varargs (fn [{:keys [dataset column-name] :as env} & args]
                 (case (count args)
                   0
                   (ds/column dataset column-name)
                   1
                   (ds/column dataset (first args)))))



(defn- apply-unary-op
  [op-kwd scalar-fn op-env op-arg]
  (cond
    (ds-col/is-column? op-arg)
    (ds-col/unary-op (ds-col/math-context op-arg) op-env op-arg op-kwd)
    (number? op-arg)
    (double (scalar-fn (double op-arg)))))


(defn register-unary-op!
  [op-kwd scalar-fn]
  (register-math-op! op-kwd :unary (partial apply-unary-op op-kwd scalar-fn)))


(register-unary-op! :log1p #(Math/log1p (double %)))
(register-unary-op! :ceil #(Math/ceil (double %)))
(register-unary-op! :floor #(Math/floor (double %)))
(register-unary-op! :sqrt #(Math/sqrt (double %)))
(register-unary-op! :- -)



(def ^:private potential-stats [:mean
                                :variance
                                :median
                                :min
                                :max
                                :skew
                                :kurtosis
                                :geometric-mean
                                :sum-of-squares
                                :sum-of-logs
                                :quadratic-mean
                                :standard-deviation
                                :population-variance
                                :sum
                                :product
                                :quartile-1
                                :quartile-3])


(doseq [stats-entry potential-stats]
  (register-math-op! stats-entry :unary-column
                     #(-> (ds-col/stats % #{stats-entry})
                          :stats-entry)))


(defn apply-binary-op
  [op-kwd scalar-fn op-env & op-args]
  (let [first-pair (take 2 op-args)
        op-args (drop 2 op-args)
        [first-arg second-arg] first-pair]
    (when-not (= 2 (count first-pair))
      (throw (ex-info "Binary operation has less that 2 operands" {})))
    (if-let [any-tensors (->> op-args
                               (filter ds-col/is-column?)
                               seq)]
      (ds-col/binary-op (ds-col/math-context (first any-tensors))
                           op-env op-args scalar-fn op-kwd)
      (apply scalar-fn op-args))))


(defn register-binary-op!
  [op-kwd scalar-fn]
  (register-math-op! op-kwd :binary
                     (partial apply-binary-op op-kwd scalar-fn)))


(register-binary-op! :+ +)
(register-binary-op! :- -)
(register-binary-op! :* *)
(register-binary-op! :/ /)
(register-binary-op! :** #(Math/pow (double %1) (double %2)))
