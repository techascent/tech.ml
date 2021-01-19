(ns tech.v3.ml.classification-test
  (:require [tech.v3.ml.classification :refer :all]
            [clojure.test :refer :all]))


(deftest test-normalized
  (is (=
       (confusion-map [:a :b :c :a] [:a :c :c :a] :none)
       {:a {:a 2}
        :c {:b 1 :c 1}}))

  (is (=
       (confusion-map [:a :b :c :a] [:a :c :c :a])
       {:a {:a 1.0}
        :c {:b 0.5 :c 0.5}})))
