(ns tech.v3.libs.smile.nlp
  (:require [clojure.string :as str]
            [pppmap.core :as ppp]
            [tech.v3.dataset :as ds])
  (:import smile.nlp.normalizer.SimpleNormalizer
           smile.nlp.stemmer.PorterStemmer
           [smile.nlp.tokenizer SimpleSentenceSplitter SimpleTokenizer]
           [smile.nlp.dictionary EnglishStopWords]
           ))


(defn resolve-stopwords [stopwords-option]
  (if (keyword? stopwords-option)
    (iterator-seq (.iterator (EnglishStopWords/valueOf (str/upper-case (name stopwords-option)))))
    stopwords-option))

(defn word-process [stemmer ^SimpleNormalizer normalizer ^String word]
  (let [word
        (-> word
            (str/lower-case)
            (#(.normalize normalizer %)))
        word (if (nil? stemmer)
               word
               (.stem stemmer word))]
    word))




(defn default-tokenize [text options]
  "Tokenizes text.
  The usage of a stemmer can be configured by options :stemmer "
  (let [normalizer (SimpleNormalizer/getInstance)
        stemmer-type (get options :stemmer :porter)
        tokenizer (SimpleTokenizer. )
        stemmer (case stemmer-type
                  :none nil
                  :porter (PorterStemmer.)
                  )
        sentence-splitter (SimpleSentenceSplitter/getInstance)
        tokens
        (->> text
             (.normalize normalizer)
             (.split sentence-splitter)
             (map #(.split tokenizer %))
             (map seq)
             flatten
             (remove nil?)
             (map #(word-process stemmer normalizer % ))
             )]
    tokens))

(defn default-text->bow [text options]
  "Converts text to token counts (a map token -> count).
   Takes options:
   `stopwords` being either a keyword naming a
   default Smile dictionary (:default :google :comprehensive :mysql)
   or a seq of stop words.
   `stemmer` being either :none or :porter for selecting the porter stemmer.
"
  (let [normalizer (SimpleNormalizer/getInstance)
        stemmer (PorterStemmer.)
        stopwords-option (:stopwords options)
        stopwords  (resolve-stopwords stopwords-option)
        processed-stop-words (map #(word-process stemmer normalizer %)  stopwords)
        tokens (default-tokenize text options)
        freqs (frequencies tokens)]
    (apply dissoc freqs processed-stop-words)))




(defn count-vectorize
  ([ds text-col bow-col text->bow-fn options]
   "Converts text column `text-col` to bag-of-words representation
   in the form of a frequency-count map"
   (ds/add-or-update-column
    ds
    (ds/new-column
     bow-col
     (ppp/ppmap-with-progress
      "text->bow"
      1000
      #(text->bow-fn % options)
      (get ds text-col)))))
  ([ds text-col bow-col text->bow-fn]
   (count-vectorize ds text-col bow-col text->bow-fn {})
   )
  )

(defn ->vocabulary-top-n [bows n]
  "Takes top-n most frequent tokens as vocabulary"
  (let [vocabulary
        (->>
         (apply merge-with + bows)
         (sort-by second)
         reverse
         (take n)
         keys)]
    vocabulary))

(defn create-vocab-all [bow ]
  "Uses all tokens as th vocabulary"
  (keys
   (apply merge bow))
  )

(defn bow->something-sparse [ds bow-col indices-col create-vocab-fn bow->sparse-fn]
  "Converts a bag-of-word column `bow-col` to a sparse data column `indices-col`.
   The exact transformation to the sparse representtaion is given by `bow->sparse-fn`"
  (let [vocabulary-list (create-vocab-fn (get ds bow-col))
        vocab->index-map (zipmap vocabulary-list  (range))
        vocabulary {:vocab vocabulary-list
                    :vocab->index-map vocab->index-map
                    :index->vocab-map (clojure.set/map-invert vocab->index-map)
                    }
        vocab->index-map (:vocab->index-map vocabulary)
        ds
        (vary-meta ds assoc
                   :count-vectorize-vocabulary vocabulary)]
    (ds/add-or-update-column
     ds
     (ds/new-column
      indices-col
      (ppp/ppmap-with-progress
       "bow->sparse"
       1000
       #(bow->sparse-fn % vocab->index-map)
       (get ds bow-col))))))




(defn tf-map [bows]
  (loop [m {} bows bows]
    (let [bow (first bows)
          token-present (zipmap (keys bow) (repeat 1))]

      (if (empty? bows)
        m
        (recur
         (merge-with + m token-present)
         (rest bows))))))


(defn idf [tf-map term bows]
  (let [n-t (count bows)
        n-d (get tf-map term)]
    (Math/log10 (/ n-t n-d ))))


(defn tf [term bow]
  (/
   (get bow term 0)
   (apply + (vals bow))))


(defn tfidf [tf-map term bow bows]
  (* (tf term bow)  (idf tf-map term bows) ))


(defn bow->tfidf [ds bow-column tfidf-column]
  "Calculates the tfidf score from bag-of-words (as token frequency maps)
   in column `bow-column` and stores them in a new column `tfid-column` as maps of token->tfidf-score."
  (let [bows (get ds bow-column)
        tf-map (tf-map bows)
        tfidf-column (ds/new-column tfidf-column
                                    (ppp/ppmap-with-progress
                                     "tfidf" 1000
                                     (fn [bow]
                                       (let [terms (keys bow)
                                             tfidfs
                                             (map
                                              #(tfidf tf-map % bow bows)
                                              terms)]
                                         (zipmap terms tfidfs)))
                                     bows))]
    (ds/add-column ds tfidf-column)))
