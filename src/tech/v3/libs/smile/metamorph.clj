(ns tech.v3.libs.smile.metamorph

  (:require
   [tech.v3.libs.smile.nlp :as nlp]
   [tech.v3.dataset :as ds]
   [pppmap.core :as ppp]
   )
  )


(defn count-vectorize [text-col bow-col text->bow-fn options]
  (fn [ctx]
    (assoc ctx :metamorph/data
           (nlp/count-vectorize (:metamorph/data ctx) text-col bow-col text->bow-fn options))))

(defn bow->tf-idf [bow-col tfidf-col]
  (fn [ctx]
    (assoc ctx :metamorph/data
           (nlp/bow->tfidf (:metamorph/data ctx) bow-col tfidf-col))))


(defn bow->something-sparse [bow-col indices-col create-vocab-fn bow->sparse-fn]
  "Converts a bag-of-word column `bow-col` to a sparse data column `indices-col`.
   The exact transformation to the sparse representtaion is given by `bow->sparse-fn`"
  (fn [ctx]
    (let [ds (:metamorph/data ctx)
          vocabulary-list (create-vocab-fn (get ds bow-col))
          vocab->index-map (zipmap vocabulary-list  (range))
          vocabulary {:vocab vocabulary-list
                      :vocab->index-map vocab->index-map
                      :index->vocab-map (clojure.set/map-invert vocab->index-map)
                      }
          vocab->index-map (:vocab->index-map vocabulary)
          ]
      (assoc ctx
             :metamorph/data
             (ds/add-or-update-column
              ds
              (ds/new-column
               indices-col
               (ppp/ppmap-with-progress
                "bow->sparse"
                1000
                #(bow->sparse-fn % vocab->index-map)
                (get ds bow-col))))
             ::count-vectorize-vocabulary vocabulary
             ))))

(defn bow->sparse-array [bow-col indices-col create-vocab-fn]
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the Maxent model.
   `vocab size` is the size of vocabluary used, sorted by token frequency "
  (bow->something-sparse bow-col indices-col create-vocab-fn nlp/bow->sparse-indices))


(defn bow->SparseArray [bow-col indices-col create-vocab-fn]
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the discrete naive bayes model. `vocab size` is the size of vocabluary used, sorted by token frequency "

  (bow->something-sparse bow-col indices-col create-vocab-fn nlp/freqs->SparseArray))
