(ns tech.v3.ml.loss
  "Simple loss functions."
  (:require [tech.v3.datatype.functional :as dfn]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.datatype.argops :as argops]))

(defn mse
  "mean squared error"
  ^double [predictions labels]
  (assert (= (count predictions) (count labels)))
  (let [n (count predictions)]
    (double
     (/ (double (-> (dfn/- predictions labels)
                    (dfn/pow 2)
                    (dfn/reduce-+)))
        n))))


(defn rmse
  "root mean squared error"
  ^double [predictions labels]
  (-> (mse predictions labels)
      (Math/sqrt)))


(defn mae
  "mean absolute error"
  ^double [predictions labels]
  (assert (= (count predictions) (count labels)))
  (let [n (count predictions)]
    (double
     (/ (double (-> (dfn/- predictions labels)
                    dfn/abs
                    (dfn/reduce-+)))
        n))))


(defn classification-accuracy
  "correct/total.
Model output is a sequence of probability distributions.
label-seq is a sequence of values.  The answer is considered correct
if the key highest probability in the model output entry matches
that label."
  ^double [lhs rhs]
  (errors/when-not-errorf
   (= (dtype/ecount lhs)
      (dtype/ecount rhs))
   "Ecounts do not match: %d %d"
   (dtype/ecount lhs) (dtype/ecount rhs))
  (/ (dtype/ecount (argops/binary-argfilter :tech.numerics/eq lhs rhs))
     (dtype/ecount lhs)))


(defn classification-loss
  "1.0 - classification-accuracy."
  ^double [lhs rhs]
  (- 1.0
     (classification-accuracy lhs rhs)))
