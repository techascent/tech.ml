(ns tech.ml.dataset.compute-math-context
  (:require [tech.compute.cpu.tensor-math :as cpu-tm]
            [tech.compute.tensor :as ct]
            [tech.ml.protocols.column :as col-proto])
  (:import [tech.compute.cpu UnaryOp BinaryOp]))



(cpu-tm/add-unary-op! :log1p (reify UnaryOp
                               (op [this val]
                                 (Math/log1p (double val)))))

(cpu-tm/add-binary-op! :** (reify BinaryOp
                             (op [this lhs rhs]
                               (Math/pow lhs rhs))))


(defrecord ComputeTensorMathContext []
  col-proto/PColumnMathContext
  (unary-op [ctx op-env op-arg op-kwd]
    (ct/unary-op! (ct/clone op-arg) 1.0 op-arg op-kwd))

  (binary-op [ctx op-env op-args op-scalar-fn op-kwd]
    (let [first-pair (take 2 op-args)
          op-args (drop 2 op-args)
          [first-arg second-arg] first-pair
          any-tensors (->> op-args
                           (filter ct/acceptable-tensor-buffer?)
                           seq)
          accumulator (ct/from-prototype (first any-tensors))]
        (if (or (ct/acceptable-tensor-buffer? first-arg)
                (ct/acceptable-tensor-buffer? second-arg))
          (ct/binary-op! accumulator 1.0 first-arg 1.0 second-arg op-kwd)
          (ct/assign! accumulator (op-scalar-fn first-arg second-arg)))
        (reduce (fn [accumulator next-arg]
                  (ct/binary-op! accumulator 1.0 accumulator 1.0 next-arg op-kwd))
                accumulator
                op-args))))
