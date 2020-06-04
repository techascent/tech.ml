(ns tech.ml.util
  (:require [tech.v2.datatype :as dtype]
            [tech.v2.datatype.typecast :as typecast]))



(defn ->str
  [item]
  (if (or (keyword? item) (symbol? item))
    (name item)
    (str item)))


(def float-ary-class (Class/forName "[F"))
(def double-ary-class (Class/forName "[D"))


(defn max-idx
  [data]
  (cond
    (instance? float-ary-class data)
    (let [^floats data data
          data-len (alength data)]
      (loop [prob-val (aget data 0)
             max-idx 0
             prob-idx 1]
        (if (< prob-idx data-len)
          (let [new-val (aget data prob-idx)
                greater? (> new-val prob-val)]
            (recur (if greater? new-val prob-val)
                   (if greater? prob-idx max-idx)
                   (unchecked-inc prob-idx)))
          max-idx)))
    (instance? double-ary-class data)
    (let [^doubles data data
          data-len (alength data)]
      (loop [prob-val (aget data 0)
             max-idx 0
             prob-idx 1]
        (if (< prob-idx data-len)
          (let [new-val (aget data prob-idx)
                greater? (> new-val prob-val)]
            (recur (if greater? new-val prob-val)
                   (if greater? prob-idx max-idx)
                   (unchecked-inc prob-idx)))
          max-idx)))
    :else
    (let [data-len (dtype/ecount data)
          data (typecast/datatype->reader :float64 data)]
      (loop [prob-val (.read data 0)
             max-idx 0
             prob-idx 1]
        (if (< prob-idx data-len)
          (let [new-val (.read data prob-idx)
                greater? (> new-val prob-val)]
            (recur (if greater? new-val prob-val)
                   (if greater? prob-idx max-idx)
                   (unchecked-inc prob-idx)))
          max-idx)))))


(defn item-at
  [data ^long idx]
  (cond
    (instance? float-ary-class data)
    (aget ^floats data idx)
    (instance? double-ary-class data)
    (aget ^doubles data idx)
    (instance? java.util.List data)
    (.get ^java.util.List data idx)
    :else
    (dtype/get-value data idx)))
