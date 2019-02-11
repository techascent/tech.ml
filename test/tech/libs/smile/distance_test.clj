(ns tech.libs.smile.distance-test
  (:require [tech.libs.smile.distance :as distance]
            [clojure.set :as c-set]
            [clojure.test :refer :all]))


(deftest all-the-things
  []
  (let [all-distances
        (->> distance/distance-metadata
             (mapcat (fn [[item-key entries]]
                       (->> entries
                            (mapcat (fn [{:keys [datatypes]}]
                                      (->> datatypes
                                           (map vector (repeat item-key))))))))
             set)
        constructible-distances (->> all-distances
                                     (filter (fn [[item-key datatype]]
                                               (try
                                                 (distance/construct-distance item-key datatype {})
                                                 (catch Throwable e
                                                   nil))))
                                     set)]
    ;;These aren't auto-construtable.  You need options or they can't be
    ;;constructed(!!-hamming)
    (is (= #{[:hamming :bit-set]
             [:hamming :int16]
             [:hamming :int16-array]
             [:hamming :int32]
             [:hamming :int32-array]
             [:hamming :int64]
             [:hamming :int8]
             [:hamming :int8-array]
             [:hamming :object-array]
             [:mahalanobis :float64-array]
             [:weighted-edit :char-array]
             [:weighted-edit :string]}
           (c-set/difference all-distances constructible-distances)))))
