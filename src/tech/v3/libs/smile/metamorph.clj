(ns tech.v3.libs.smile.metamorph

  (:require
   [tech.v3.libs.smile.nlp :as nlp]
   [tech.v3.dataset :as ds]
   [pppmap.core :as ppp]))


(defn count-vectorize [text-col bow-col text->bow-fn options]
  (fn [ctx]
    (assoc ctx :metamorph/data
           (nlp/count-vectorize (:metamorph/data ctx) text-col bow-col text->bow-fn options))))


(defn bow->something-sparse [bow-col indices-col create-vocab-fn bow->sparse-fn]
  "Converts a bag-of-word column `bow-col` to a sparse data column `indices-col`.
   The exact transformation to the sparse representtaion is given by `bow->sparse-fn`"
  (fn [ctx]
    (let [{:keys [ds vocab]} (nlp/bow->sparse-and-vocab (:metamorph/data ctx)
                                            bow-col indices-col
                                            create-vocab-fn bow->sparse-fn)]
      (assoc ctx :metamorph/data
             ds ::count-vectorize-vocabulary vocab))))

(defn bow->sparse-array [bow-col indices-col create-vocab-fn]
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the Maxent model.
   `vocab size` is the size of vocabluary used, sorted by token frequency "
  (bow->something-sparse bow-col indices-col create-vocab-fn nlp/bow->sparse-indices))


(defn bow->SparseArray [bow-col indices-col create-vocab-fn]
  "Converts a bag-of-word column `bow-col` to sparse indices column `indices-col`,
   as needed by the discrete naive bayes model. `vocab size` is the size of vocabluary used, sorted by token frequency "
  (bow->something-sparse bow-col indices-col create-vocab-fn nlp/freqs->SparseArray))

(defn bow->tfidf [bow-column tfidf-column]
  "Calculates the tfidf score from bag-of-words (as token frequency maps)
   in column `bow-column` and stores them in a new column `tfid-column` as maps of token->tfidf-score."
  (fn [ctx]
    (assoc ctx :metamorph/data
           (nlp/bow->tfidf
            (:metamorph/data ctx)
            bow-column
            tfidf-column))))
