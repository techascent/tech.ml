(ns tech.libs.smile.kernel-test
  (:require [tech.libs.smile.kernels :as kernels]
            [clojure.test :refer :all])
  (:import [smile.math.kernel MercerKernel]))

(deftest construct-the-things
  (->> kernels/kernels
       (mapcat (fn [[k v]]
                 (map (fn [v]
                        [k (:datatype v)])
                      v)))
       (map (fn [[kernel-type datatype]]
              (let [kernel (kernels/construct kernel-type datatype {})]
                (is (instance? MercerKernel kernel)))))
       doall))
