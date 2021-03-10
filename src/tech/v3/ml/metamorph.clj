(ns tech.v3.ml.metamorph
  (:require [tech.v3.ml :as ml]
            [tech.v3.libs.smile.nlp :as nlp]
            [tech.v3.libs.smile.discrete-nb]
            [tech.v3.libs.smile.maxent]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.libs.smile.metamorph :as smile-mm]
            ))

(defn model [options]
  (fn [{:metamorph/keys [id data mode] :as ctx}]
    (case mode
      :fit (assoc ctx id (ml/train data  options))
      :transform  (assoc ctx :metamorph/data (ml/predict data (get ctx id))))))


(comment

  (defn pipeline
    [& ops]
    (let [ops-with-id (map-indexed vector ops)] ;; add consecutive number to each operation
      (fn local-pipeline
        ([] (local-pipeline {})) ;; can be called without a context
        ([ctx]
         (let [ctx (if-not (map? ctx)
                     {:metamorph/data ctx} ctx)] ;; if context is not a map, pack it to the map
           (dissoc (reduce (fn [curr-ctx [id op]] ;; go through operations
                             (if (keyword? op)    ;; bare keyword means a mode!
                               (assoc curr-ctx :metamorph/mode op) ;; set current mode
                               (-> curr-ctx
                                   (assoc :metamorph/id id) ;; assoc id of the operation
                                   (op)                     ;; call it
                                   (dissoc :metamorph/id))))      ;; dissoc id
                           ctx ops-with-id) :metamorph/mode)))))) ;; dissoc mode
  (defn model-maxent [options]
    (fn [ctx]
      ((model (merge options
                     {:p (count (-> ctx :tech.v3.libs.smile.metamorph/count-vectorize-vocabulary :vocab->index-map))})) ctx))
    )

  (def pipe
    (pipeline
     (fn [ctx]
       (assoc ctx :metamorph/data
              (ds/select-columns (:metamorph/data ctx) [:Text :Score])))
        
     (smile-mm/count-vectorize :Text :bow nlp/default-text->bow {})
     (smile-mm/bow->sparse-array :bow :bow-sparse #(nlp/->vocabulary-top-n % 1000))
     (fn [ctx]
       (assoc ctx :metamorph/data
              (ds-mod/set-inference-target  (:metamorph/data ctx) :Score)))
     (model-maxent {:model-type :maxent-multinomial
                     :sparse-column :bow-sparse})

     
     ))

  (def trained-pipeline
    (pipe
     {:metamorph/mode :fit
      :metamorph/data
      (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword })}))

  (def predicted-pipeline
    (pipe
     (merge
      trained-pipeline
      {:metamorph/mode :transform
       :metamorph/data
       (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword })})))


  )
