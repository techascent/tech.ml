(ns tech.libs.smile.distance
  (:require [clojure.reflect :refer [reflect]]
            [tech.libs.smile.utils :as utils]
            [camel-snake-kebab.core :refer [->kebab-case]])
  (:import [smile.math.distance Distance Metric]))


(def package-name "smile.math.distance")


(def java-classes
  #{"ChebyshevDistance"
    "CorrelationDistance"
    "DynamicTimeWarping"
    "EditDistance"
    "EuclideanDistance"
    "HammingDistance"
    "JaccardDistance"
    "JensenShannonDistance"
    "LeeDistance"
    "MahalanobisDistance"
    "ManhattanDistance"
    "MinkowskiDistance"
    "SparseChebyshevDistance"
    "SparseEuclideanDistance"
    "SparseManhattanDistance"
    "SparseMinkowskiDistance"})


(defn reflect-cls [cls]
  (reflect (utils/class-name->class package-name cls)))

(defn- trim-distance
  [^String str-data]
  (let [distance-idx (.indexOf str-data "Distance")]
    (if (> distance-idx 0)
      (.substring str-data 0 distance-idx)
      str-data)))


(defn- class-name->distance-type
  [^String cls-name]
  (-> (trim-distance cls-name)
      ->kebab-case
      keyword))


(defn generate-distance-metadata
  []
  (->> java-classes
       (map
        (fn [java-cls-name]
          (let [item-kwd (class-name->distance-type java-cls-name)
                reflect-data (reflect-cls java-cls-name)]
            [item-kwd {:class-name java-cls-name
                       :name item-kwd
                       :datatypes (utils/method-datatype "d" reflect-data)}])))
       (into {})))


(def distance-metadata
  {:chebyshev [{:class-name "ChebyshevDistance"
                :name :chebyshev
                :datatypes #{:float32-array :float64-array :int32-array}}
                {:class-name "SparseChebyshevDistance"
                 :name :sparse-chebyshev
                 :datatypes #{:sparse}}]
   :correlation [{:class-name "CorrelationDistance"
                   :name :correlation
                   :datatypes #{:float64-array}}]
   :dynamic-time-warping [{:class-name "DynamicTimeWarping"
                            :name :dynamic-time-warping
                            :datatypes #{:float32-array
                                         :float64-array
                                         :int32-array}
                            :options [{:type :distance
                                       :name :distance
                                       :default {:distance-type :euclidean}}
                                      {:type :float64
                                       :name :radius
                                       :default 0.5
                                       :range :fraction}]}]
   :edit [{:class-name "EditDistance"
            :name :edit
            :datatypes #{:char-array :string}
            :options [{:name :max-string-length
                       :type :int32
                       :default 512}
                      {:name :damerau
                       :type :boolean
                       :default false}]}]
   :weighted-edit [{:class-name "EditDistance"
                     :name :weighted-edit
                     :datatypes #{:char-array :string}
                     :options [{:name :weight
                                :type :float64-array-array}
                               {:name :radius
                                :type :float64
                                :range :fraction}]}]
   :euclidean [{:class-name "EuclideanDistance"
                 :name :euclidean
                 :datatypes #{:float32-array
                                   :float64-array
                                   :int32-array}
                 :options [{:name :weight
                            :type :float64-array
                            :attributes #{:optional?}}]}
               {:class-name "SparseEuclideanDistance"
                :name :sparse-euclidean
                :datatypes #{:sparse}
                :options [{:name :weight
                           :type :float64-array
                           :attributes #{:optional?}}]}]
   :hamming [{:class-name "HammingDistance"
               :name :hamming
               :datatypes #{:bit-set
                            :int16
                            :int16-array
                            :int32
                            :int32-array
                            :int64
                            :int8
                            :int8-array
                            :object-array}}]
   :jaccard [{:class-name "JaccardDistance" :name :jaccard :datatypes #{:java.util.Set :object-array}}]
   :jensen-shannon [{:class-name "JensenShannonDistance"
                      :name :jensen-shannon
                      :datatypes #{:float64-array}}]
   :lee [{:class-name "LeeDistance"
           :name :lee
           :datatypes #{:int32-array}
           :options [{:name :q-ary-alphabet-size
                      :type :int32
                      :default 20
                      :range :>1}]}]
   :mahalanobis [{:class-name "MahalanobisDistance"
                   :name :mahalanobis
                   :datatypes #{:float64-array}
                   :options [{:name :covariance-matrix
                              :type :float64-array-array}]}]
   :manhattan [{:class-name "ManhattanDistance"
                 :name :manhattan
                :datatypes #{:float32-array :float64-array :int32-array}
                :options [{:name :weight
                           :type :float64-array
                           :attributes #{:optional?}}]}
               {:class-name "SparseManhattanDistance"
                :name :sparse-manhattan
                :datatypes #{:sparse}
                :options [{:name :weight
                           :type :float64-array
                           :attributes #{:optional?}}]}]

   :minkowski [{:class-name "MinkowskiDistance"
                 :name :minkowski
                 :datatypes #{:float32-array :float64-array :int32-array}
                 :options [{:name :p
                            :type :int32
                            :range :>0
                            :default 4}
                           {:name :weight
                            :type :float64-array
                            :attributes #{:optional?}}]}
               {:class-name "SparseMinkowskiDistance"
                :name :sparse-minkowski
                :datatypes #{:sparse}
                :options [{:name :p
                            :type :int32
                            :range :>0
                            :default 4}
                           {:name :weight
                            :type :float64-array
                            :attributes #{:optional?}}]}]})


(defmulti distance-type->metadata
  (fn [distance-type datatype]
    distance-type))


(defmethod distance-type->metadata :default
  [distance-type datatype]
  (if-let [entries (get distance-metadata distance-type)]
    (if-let [retval (->> entries
                         (filter (fn [{:keys [datatypes]}]
                                   (contains? datatypes datatype)))
                         first)]
      retval
      (throw (ex-info "Matching entries found but specific datatype missing"
                      {:distance-type distance-type
                       :datatype datatype
                       :entries entries})))
    (throw (ex-info "Unrecognized distance type"
                    {:distance-type distance-type
                     :possible-types (keys distance-metadata)}))))


(defmethod utils/option->class-type :distance
  [& args]
  Distance)


(defn construct-distance
  [distance-type datatype options]
  (-> (distance-type->metadata distance-type datatype)
      (assoc :datatype datatype)
      (utils/construct package-name options)))


(defmethod utils/option-value->value :distance
  [class-metadata meta-option option-value]
  (construct-distance (:distance-type option-value)
                      (or (:datatype option-value) :float64-array)
                      option-value))
