(ns tech.v3.libs.smile.regression
  "Namespace to require to enable a set of smile regression models"
  (:require [tech.v3.ml :as ml]
            [tech.v3.ml.model :as model]
            [tech.v3.dataset :as ds]
            [tech.v3.tensor :as dtt]
            [tech.v3.dataset.utils :as ds-utils]
            [tech.v3.datatype :as dtype]
            [tech.v3.ml.gridsearch :as ml-gs]
            [tech.v3.libs.smile.data :as smile-data]
            [tech.v3.libs.smile.protocols :as smile-proto])
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
           [smile.data.formula Formula TechFactory Variable]
           [smile.data DataFrame]
           [java.lang.reflect Field]
           [java.util Properties List]))


(set! *warn-on-reflection* true)


(def ^:private cart-loss-table
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
  [^LinearModel thawed-model ds]
  (let [^List val-rdr (ds/value-reader ds)]
    (->> (dtype/make-reader
          :float64
          (ds/row-count ds)
          (.predict thawed-model
                    ^doubles (dtype/->double-array (val-rdr idx))))
         (dtype/make-container :java-array :float64))))


(defn- predict-df
  [^DataFrameRegression thawed-model ds]
  (let [df (smile-data/dataset->smile-dataframe ds)]
    (smile-proto/initialize-model-formula! thawed-model ds)
    (.predict thawed-model df)))


(def ^:private regression-metadata
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
                 :gridsearch-options {:lambda1 (ml-gs/linear 1e-2 1e2)
                                      :lambda2 (ml-gs/linear 1e-4 1e2)
                                      :tolerance (ml-gs/linear 1e-6 1e-2)
                                      :max-iterations (ml-gs/linear 1e4 1e7)}
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
    :gridsearch-options {:lambda (ml-gs/linear 1e-4 1e1)
                         :tolerance (ml-gs/linear 1e-6 1e-2)
                         :max-iterations (ml-gs/linear 1e4 1e7 100 :int64)}
    :property-name-stem "smile.lasso"
    :constructor #(LASSO/fit ^Formula %1 ^DataFrame %2 ^Properties %3)
    :predictor predict-linear-model
    }


   :ridge
   {:options [{:name :lambda
               :type :float64
               :default 1.0
               :range :>0}]
    :gridsearch-options {:lambda (ml-gs/linear 1e-4 1e4)}
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
    :predictor predict-df}
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
    :predictor predict-df}
   })


(defmulti ^:private model-type->regression-model
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


(defn- train
  [feature-ds label-ds options]
  (let [entry-metadata (model-type->regression-model
                        (model/options->model-type options))
        target-colnames (ds/column-names label-ds)
        feature-colnames (ds/column-names feature-ds)
        _ (when-not (= 1 (count target-colnames))
            (throw (Exception. "Dataset has none or too many target columns.")))
        formula (smile-proto/make-formula
                 (ds-utils/column-safe-name (first target-colnames))
                 (map ds-utils/column-safe-name feature-colnames))
        full-ds (merge feature-ds label-ds)
        data (smile-data/dataset->smile-dataframe full-ds)
        properties (smile-proto/options->properties entry-metadata full-ds options)
        ctor (:constructor entry-metadata)
        model (ctor formula data properties)]
    (model/model->byte-array model)))


(defn- thaw
  [model-data]
  (model/byte-array->model model-data))


(defn- predict
  [feature-ds thawed-model {:keys [target-columns options]}]
  (let [entry-metadata (model-type->regression-model
                        (model/options->model-type options))
        predictor (:predictor entry-metadata)
        target-cname (first target-columns)]
    (-> (predictor thawed-model feature-ds)
        (dtype/clone)
        (dtt/->tensor)
        (model/finalize-regression target-cname))))


(defn- explain
  [thawed-model {:keys [feature-columns]} _options]
  (when (instance? LinearModel thawed-model)
    (let [^LinearModel model thawed-model
          weights (.coefficients model)
          bias (.intercept model)]
      {:bias bias
       :coefficients (->> (map vector
                               feature-columns
                               (dtype/->reader weights))
                          (sort-by (comp second) >))})))


(doseq [[reg-kwd reg-def] regression-metadata]
  (ml/define-model! (keyword "smile.regression" (name reg-kwd))
    train predict {:thaw-fn thaw
                   :explain-fn explain
                   :hyperparameters (:gridsearch-options reg-def)}))


(comment
  (do
    (require '[tech.v3.dataset.column-filters :as cf])
    (require '[tech.v3.dataset.modelling :as ds-mod])
    (require '[tech.v3.ml.loss :as loss])
    (def src-ds (ds/->dataset "test/data/iris.csv"))
    (def ds (->  src-ds
                 (ds/categorical->number cf/categorical)
                 (ds-mod/set-inference-target "species")))
    (def feature-ds (cf/feature ds))
    (def split-data (ds-mod/train-test-split ds))
    (def train-ds (:train-ds split-data))
    (def test-ds (:test-ds split-data))
    (def model (ml/train train-ds {:model-type :smile.regression/elastic-net}))
    (def prediction (ml/predict test-ds model))
    )
  )
