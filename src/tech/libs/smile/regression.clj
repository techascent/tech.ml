(ns tech.libs.smile.regression
  (:require [tech.ml.model :as model]
            [tech.ml.protocols.system :as ml-proto]
            [tech.ml.dataset :as ds]
            [tech.ml.registry :as registry]
            ;;Kernels have to be loaded
            ;;[tech.libs.smile.kernels]
            [tech.v2.datatype :as dtype]
            [tech.ml.gridsearch :as ml-gs]
            [tech.ml.util :as ml-util]
            [tech.libs.smile.protocols :as smile-proto])
  (:import [smile.regression
            Regression
            DataFrameRegression
            GradientTreeBoost
            RandomForest
            ;; NeuralNetwork$ActivationFunction
            RidgeRegression
            ElasticNet
            LASSO
            LinearModel
            ;; OnlineRegression
            ]
           [smile.data.formula Formula]
           [smile.data DataFrame]
           [java.lang.reflect Field]
           [java.util Properties List]))


(set! *warn-on-reflection* true)


(def cart-loss-table
  {
   ;; Least squares regression. Least-squares is highly efficient for
   ;; normally distributed errors but is prone to long tails and outliers.
   :least-squares "LeastSquares"
   ;; Quantile regression. The gradient tree boosting based
   ;; on this loss function is highly robust. The trees use only order
   ;; information on the input variables and the pseudo-response has only
   ;; two values {-1, +1}. The line searches (terminal node values) use
   ;; only specified quantile ratio.
   :quantile "Quantile"
   ;; Least absolute deviation regression. The gradient tree boosting based
   ;; on this loss function is highly robust. The trees use only order
   ;; information on the input variables and the pseudo-response has only
   ;; two values {-1, +1}. The line searches (terminal node values) use
   ;; only medians. This is a special case of quantile regression of q = 0.5.
   :least-absolute-deviation "LeastAbsoluteDeviation"
   ;; Huber loss function for M-regression, which attempts resistance to
   ;; long-tailed error distributions and outliers while maintaining high
   ;; efficency for normally distributed errors.
   :huber "Huber"})


(defn- predict-linear-model
  [^LinearModel thawed-model ds options]
  (let [^List row-major-dataset (ds/->row-major ds options)]
    (->> (dtype/make-reader
          :float64
          (ds/row-count ds)
          (.predict thawed-model
                    ^doubles (:features (.get row-major-dataset idx))))
         (dtype/make-container :java-array :float64))))


(defn- predict-tuple
  [^Regression thawed-model ds options]
  (let [df (ds/dataset->smile-dataframe ds)]
    (smile-proto/initialize-model-formula! thawed-model options)
    (->> (dtype/make-reader
          :float64
          (ds/row-count ds)
          (.predict thawed-model (.get df idx)))
         (dtype/make-container :java-array :float64))))


;;This currently fails because of smile issue
;;https://github.com/haifengl/smile/issues/554
(defn- predict-df
  [^DataFrameRegression thawed-model ds options]
  (let [df (ds/dataset->smile-dataframe ds)]
    (smile-proto/initialize-model-formula! thawed-model options)
    (.predict thawed-model df)))


(def regression-metadata
  {:elastic-net {:options [{:name :lambda1
                            :type :float64
                            :default 0.1
                            :range :>0}
                           {:name :lambda2
                            :type :float64
                            :default 0.1
                            :range :>0}

                           {:name :tolerance
                            :type :float64
                            :default 1e-4
                            :range :>0}

                           {:name :max-iterations
                            :type :int32
                            :default (int 1000)
                            :range :>0}]
                 :gridsearch-options {:lambda1 (ml-gs/exp [1e-2 1e2])
                                      :lambda2 (ml-gs/exp [1e-4 1e2])
                                      :tolerance (ml-gs/exp [1e-6 1e-2])
                                      :max-iterations (ml-gs/exp [1e4 1e7])}
                 :property-name-stem "smile.elastic.net"
                 :constructor #(ElasticNet/fit %1 %2 %3)
                 :predictor predict-linear-model}


   :lasso
   {:options [{:name :lambda
               :type :float64
               :default 1.0
               :range :>0}
              {:name :tolerance
               :type :float64
               :default 1e-4
               :range :>0}
              {:name :max-iterations
               :type :int32
               :default 1000
               :range :>0}]
    :gridsearch-options {:lambda (ml-gs/exp [1e-4 1e1])
                         :tolerance (ml-gs/exp [1e-6 1e-2])
                         :max-iterations (ml-gs/linear-long [1e4 1e7])}
    :property-name-stem "smile.lasso"
    :constructor #(LASSO/fit ^Formula %1 ^DataFrame %2 ^Properties %3)
    :predictor predict-linear-model
    }


   :ridge
   {:options [{:name :lambda
               :type :float64
               :default 1.0
               :range :>0}]
    :gridsearch-options {:lambda (ml-gs/exp [1e-4 1e4])}
    :property-name-stem "smile.ridge"
    :constructor #(RidgeRegression/fit ^Formula %1 ^DataFrame %2 ^Properties %3)
    :predictor predict-linear-model}


   :gradient-tree-boost
   {:options [{:name :trees
               :type :int32
               :default 500
               :range :>0}
              {:name :loss
               :type :enumeration
               :lookup-table cart-loss-table
               :default :least-absolute-deviation}
              {:name :max-depth
               :type :int32
               :default 20
               :range :>0}
              {:name :max-nodes
               :type :int32
               :default 6
               :range :>0}
              {:name :node-size
               :type :int32
               :default 5
               :range :>0}
              {:name :shrinkage
               :type :float64
               :default 0.05
               :range :>0}
              {:name :sample-rate
               :type :float64
               :default 0.7
               :range [0.0 1.0]}]
    :property-name-stem "smile.bgt.trees"
    :constructor #(GradientTreeBoost/fit %1 %2 %3)
    :predictor predict-tuple}
   :random-forest
   {:options [
              {:name :trees
               :type :int32
               :default 500
               :range :>0}

              {:name :max-depth
               :type :int32
               :default 20
               :range :>0}

              {:name :max-nodes
               :type :int32
               :default #(unchecked-int (max 5 (/ (ds/row-count %) 5)))
               :range :>0}

              {:name :node-size
               :type :int32
               :default 5
               :range :>0}

              {:name :sample-rate
               :type :float64
               :default 1.0
               :range [0.0 1.0]}

              ]
    :property-name-stem "smile.random.forest"
    :constructor #(RandomForest/fit %1 %2 %3)
    :predictor predict-tuple}
   })


(defmulti model-type->regression-model
  (fn [model-type]
    model-type))


(defmethod model-type->regression-model :default
  [model-type]
  (if-let [retval (get regression-metadata model-type)]
    retval
    (throw (ex-info "Failed to find regression model"
                    {:model-type model-type}))))


(defmethod model-type->regression-model :regression
  [model-type]
  (get regression-metadata :elastic-net))



(defrecord SmileRegression []
  ml-proto/PMLSystem
  (system-name [_] :smile.regression)
  (gridsearch-options [system options]
    (let [entry-metadata (model-type->regression-model
                          (model/options->model-type options))]
      (if-let [retval (:gridsearch-options entry-metadata)]
        retval
        (throw (ex-info "Model type does not support auto gridsearch yet"
                        {:entry-metadata entry-metadata})))))
  (train [system options dataset]
    (let [entry-metadata (model-type->regression-model
                          (model/options->model-type options))
          target-colnames (->> (map meta dataset)
                               (filter #(= :inference (:column-type %))))
          _ (when-not (= 1 (count target-colnames))
              (throw (Exception. "Dataset has none or too many target columns.")))
          formula (Formula. (ml-util/->str (:name (first target-colnames))))
          data (ds/dataset->smile-dataframe dataset)
          properties (smile-proto/options->properties entry-metadata dataset options)
          ctor (:constructor entry-metadata)
          model (ctor formula data properties)]
      (model/model->byte-array model)))
  (thaw-model [system model]
    (model/byte-array->model model))
  (predict [system options thawed-model dataset]
    (let [entry-metadata (model-type->regression-model
                          (model/options->model-type options))
          predictor (:predictor entry-metadata)]
      (predictor thawed-model dataset options))))


(defn get-field
  [obj fname]
  (let [field (doto (.getDeclaredField (.getClass ^Object obj) fname)
                (.setAccessible true))]
    (.get field obj)))


(defn explain-linear-model
  [model {:keys [feature-columns]}]
  (let [weights (get-field model "w")
        bias (get-field model "b")]
    {:bias bias
     :coefficients (->> (map vector
                             feature-columns
                             (dtype/->reader weights))
                        (sort-by (comp second) >))}))


(extend-protocol ml-proto/PInternalMLModelExplain
  LASSO
  (model-explain-model [model options]
    (explain-linear-model model options))
  RidgeRegression
  (model-explain-model [model options]
    (explain-linear-model model options))
  ElasticNet
  (model-explain-model [model options]
    (explain-linear-model model options)))


(def system (constantly (->SmileRegression)))


(registry/register-system (system))
