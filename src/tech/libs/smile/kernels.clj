(ns tech.libs.smile.kernels
  (:require [clojure.reflect :refer [reflect]]
            [camel-snake-kebab.core :refer [->kebab-case]]
            [tech.datatype :as dtype]
            [tech.libs.smile.utils :as utils])
  (:import [smile.math.kernel MercerKernel]
           [java.lang.reflect Constructor]))


(def package-name "smile.math.kernel")

(def kernel-class-names
  #{"BinarySparseGaussianKernel"
    "BinarySparseHyperbolicTangentKernel"
    "BinarySparseLaplacianKernel"
    "BinarySparseLinearKernel"
    "BinarySparsePolynomialKernel"
    "BinarySparseThinPlateSplineKernel"
    "GaussianKernel"
    "HellingerKernel"
    "HyperbolicTangentKernel"
    "LaplacianKernel"
    "LinearKernel"
    "PearsonKernel"
    "PolynomialKernel"
    "SparseGaussianKernel"
    "SparseHyperbolicTangentKernel"
    "SparseLaplacianKernel"
    "SparseLinearKernel"
    "SparsePolynomialKernel"
    "SparseThinPlateSplineKernel"
    "ThinPlateSplineKernel"
    })


(def kernel-metadata
  {:gaussian {:options [{:name :sigma
                         :type :float64
                         :range :>0
                         :default 2.0}]}

   :hyperbolic-tangent {:options [{:name :scale
                                   :type :float64
                                   :range :>0
                                   :default 1.0}
                                  {:name :offset
                                   :type :float64
                                   :default 0.0}]}

   :laplacian {:options [{:name :sigma
                          :type :float64
                          :range :>0
                          :default 2.0}]}

   :linear {}

   :polynomial {:options [{:name :degree
                           :type :int32
                           :range :>1
                           :default 2}
                          {:name :scale
                           :type :float64
                           :range :>1
                           :default 1.0}
                          {:name :offset
                           :type :float64
                           :default 0.0}]}

   :thin-plate-spline {:options [{:name :sigma
                                  :type :float64
                                  :range :>0
                                  :default 2.0}]}

   :pearson {:options [{:name :omega
                        :type :float64
                        :default 1.0}
                       {:name :sigma
                        :type :float64
                        :default 1.0}]}
   :hellinger {}})


(defn- reflect-kernel
  [kernel-name]
  (reflect (Class/forName (str package-name "." kernel-name))))


(defn- datatype
  [reflect-data]
  (let [dtype (utils/method-datatype "k" reflect-data)]
    (when-not (= 1 (count dtype))
      (throw (ex-info "Kernel supports more than one datatype"
                      {:dtype dtype})))
    (first dtype)))


(def kernels
  (->> kernel-class-names
       (mapv (fn [nm]
               (let [reflect-info (reflect-kernel nm)
                     kernel-name (keyword (->kebab-case nm))]
                {:name kernel-name
                 :class-name nm
                 :type (->> (keys kernel-metadata)
                            (filter #(.contains (name kernel-name) (name %)))
                            first)
                 :datatype (datatype reflect-info)})))
       (group-by :type)))


(defmulti find-kernel
  (fn [kernel-type datatype]
    [kernel-type datatype]))


(defmethod find-kernel :default
  [kernel-type datatype]
  (if-let [retval (->> (get kernels kernel-type)
                       (filter #(= datatype (:datatype %)))
                       first)]
    retval
    (throw (ex-info "Failed to find kernel"
                    {:kernel-type kernel-type
                     :datatype datatype
                     :available (->> kernels
                                     (map (fn [[k v]]
                                            [k (->> v
                                                    (map :datatype)
                                                    set)])))}))))

(defn dense-kernels
  []
  (->> kernels
       (filterv #(contains? (:datatype %) :float64-array))))

(defn sparse-binary-kernels
  []
  (->> kernels
       (filterv #(contains? (:datatype %) :int32-array))))

(defn sparse-kernels
  []
  (->> kernels
       (filterv #(contains? (:datatype %) :sparse))))


(defn construct
  ^MercerKernel [kernel-type datatype options]
  (utils/construct (assoc (merge (get kernel-metadata kernel-type)
                                 (find-kernel kernel-type datatype))
                          :datatype datatype)
                   package-name
                   options))


(defmethod utils/option->class-type :mercer-kernel
  [& args]
  MercerKernel)


(defmethod utils/option-value->value :mercer-kernel
  [class-metadata meta-option option-value]
  (construct (:kernel-type option-value)
             (or (:datatype option-value)
                 (:datatype class-metadata)
                 :float64-array)
             option-value))
